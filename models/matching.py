import paddle
import paddle.nn as nn
from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(nn.Layer):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self):
        super().__init__()
        self.superpoint = SuperPoint(weights_path="models/weights/superpoint_v1.pdparams")
        self.superpoint.eval()
        self.superglue = SuperGlue(weights_path="models/weights/superglue_outdoor.pdparams")
        self.superglue.eval()

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ["keypoints0", "keypoints1"] exist in input
        Args:
          data: dictionary with minimal keys: ["image0", "image1"]
        """
        pred = {}
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = paddle.stack(data[k])
        # Perform the matching
        pred = {**pred, **self.superglue(data)}
        return pred
