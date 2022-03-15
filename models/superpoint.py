import os.path as osp
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def simple_nms(scores, nms_radius):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        if len(x.shape) == 3:
            N, H, W = x.shape
            x = x.reshape((N, 1, H, W))
        mp = F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
        return mp[:, 0 ,: ,:]

    zeros = paddle.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.astype("float32")) > 0
        supp_scores = paddle.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return paddle.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border, height, width):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = paddle.topk(scores, k, axis=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s=8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= paddle.to_tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)]).astype(keypoints.dtype)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    descriptors = F.grid_sample(
        descriptors, keypoints.reshape((b, 1, -1, 2)))
    descriptors = F.normalize(
        descriptors.reshape((b, c, -1)), p=2, axis=1)
    return descriptors


class SuperPoint(nn.Layer):
  """SuperPoint Convolutional Detector and Descriptor
  SuperPoint: Self-Supervised Interest Point Detection and
  Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
  Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
  """
  def __init__(self, weights_path="", 
                     nms_radius=4,
                     keypoint_threshold=0.005,
                     max_keypoints=-1):
    super(SuperPoint, self).__init__()
    self.nms_radius = nms_radius
    self.keypoint_threshold = keypoint_threshold
    self.remove_borders = 4
    self.max_keypoints = max_keypoints
    # net
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = nn.Conv2D(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = nn.Conv2D(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = nn.Conv2D(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = nn.Conv2D(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = nn.Conv2D(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = nn.Conv2D(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = nn.Conv2D(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = nn.Conv2D(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = nn.Conv2D(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = nn.Conv2D(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = nn.Conv2D(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = nn.Conv2D(c5, d1, kernel_size=1, stride=1, padding=0)
    # load
    if osp.exists(weights_path):
        self.set_state_dict(paddle.load(weights_path))
        print("Loaded SuperPoint model.")

  def forward(self, data):
    """ Compute keypoints, scores, descriptors for image """
    # Shared Encoder
    x = self.relu(self.conv1a(data["image"]))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Compute the dense keypoint scores
    cPa = self.relu(self.convPa(x))
    scores = self.convPb(cPa)
    scores = F.softmax(scores, 1)[:, :-1]
    b, _, h, w = scores.shape
    scores = scores.transpose((0, 2, 3, 1)).reshape((b, h, w, 8, 8))
    scores = scores.transpose((0, 1, 3, 2, 4)).reshape((b, h * 8, w * 8))
    scores = simple_nms(scores, self.nms_radius)
    # Extract keypoints
    keypoints = [
        paddle.nonzero(s > self.keypoint_threshold) for s in scores]
    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, self.remove_borders, h * 8, w * 8)
        for k, s in zip(keypoints, scores)]))
    # Keep the k keypoints with highest score
    if self.max_keypoints >= 0:
        keypoints, scores = list(zip(*[
            top_k_keypoints(k, s, self.remove_borders)
            for k, s in zip(keypoints, scores)]))
    # Convert (h, w) to (x, y)
    keypoints = [paddle.flip(k, [1]).astype("float32") for k in keypoints]
    # Compute the dense descriptors
    cDa = self.relu(self.convDa(x))
    descriptors = self.convDb(cDa)
    descriptors = F.normalize(descriptors, p=2, axis=1)
    # Extract descriptors
    descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, descriptors)]
    return {
        "keypoints": keypoints,
        "scores": scores,
        "descriptors": descriptors,
    }