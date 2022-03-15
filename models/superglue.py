from copy import deepcopy
import os.path as osp
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def MLP(channels, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1D(channels[i - 1], channels[i], kernel_size=1, bias_attr=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1D(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = paddle.to_tensor(1).astype(kpts.dtype)
    size = paddle.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True) * 0.7
    return (kpts - center[:, None, :].squeeze(-1)) / scaling[:, None, :].squeeze(-1)


class KeypointEncoder(nn.Layer):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.initializer.Constant(value=0.0)(self.encoder[-1].bias)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose((0, 2, 1)), scores.unsqueeze(1)]
        return self.encoder(paddle.concat(inputs, axis=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = paddle.einsum("bdhn,bdhm->bhnm", query, key) / dim ** .5
    prob = F.softmax(scores, axis=-1)
    return paddle.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Layer):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1D(d_model, d_model, kernel_size=1)
        self.proj = nn.LayerList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.shape[0]
        query, key, value = [l(x).reshape((batch_dim, self.dim, self.num_heads, -1))
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.reshape((batch_dim, self.dim*self.num_heads, -1)))


class AttentionalPropagation(nn.Layer):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.initializer.Constant(value=0.0)(self.mlp[-1].bias)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(paddle.concat([x, message], axis=1))


class AttentionalGNN(nn.Layer):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.LayerList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == "self":
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = paddle.zeros_like(log_mu), paddle.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - paddle.logsumexp(Z + v.unsqueeze(1), axis=2)
        v = log_nu - paddle.logsumexp(Z + u.unsqueeze(2), axis=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = paddle.to_tensor(1).astype(scores.dtype)
    ms, ns = (m * one).astype(scores.dtype), (n * one).astype(scores.dtype)
    bins0 = alpha.expand((b, m, 1))
    bins1 = alpha.expand((b, 1, n))
    alpha = alpha.expand((b, 1, 1))
    couplings = paddle.concat([paddle.concat([scores, bins0], -1),
                               paddle.concat([bins1, alpha], -1)], 1)
    norm = - (ms + ns).log()
    log_mu = paddle.concat([norm.expand([m]).unsqueeze(-1), ns.log()[None] + norm])
    log_nu = paddle.concat([norm.expand([n]).unsqueeze(-1), ms.log()[None] + norm])
    log_mu = log_mu[None].expand((b, -1, 1))
    log_nu = log_nu[None].expand((b, -1, 1)).transpose((0, 2, 1))
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return paddle.to_tensor(x.shape[dim]).astype(x.dtype).cumsum(0) - 1


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


class SuperGlue(nn.Layer):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """
    def __init__(self, weights_path="", sinkhorn_iterations=20, match_threshold=0.2):
        super().__init__()
        self.kenc = KeypointEncoder(256, [32, 64, 128, 256])
        self.gnn = AttentionalGNN(feature_dim=256, layer_names=["self", "cross"] * 9)
        self.final_proj = nn.Conv1D(256, 256, kernel_size=1, bias_attr=True)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        bin_score = paddle.create_parameter([1], dtype="float32", 
                    default_initializer=nn.initializer.Constant(value=1.))
        self.add_parameter("bin_score", bin_score)
        if osp.exists(weights_path):
            self.set_state_dict(paddle.load(str(weights_path)))
            print("Loaded SuperGlue model.")

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                "matches0": -paddle.ones_like(shape0, dtype="int64"),
                "matches1": -paddle.ones_like(shape1, dtype="int64"),
                "matching_scores0": paddle.zeros_like(shape0, dtype=kpts0.dtype),
                "matching_scores1": paddle.zeros_like(shape1, dtype=kpts1.dtype)
            }
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data["image0"].shape)
        kpts1 = normalize_keypoints(kpts1, data["image1"].shape)
        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Compute matching descriptor distance.
        scores = paddle.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / 256 ** .5
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.sinkhorn_iterations)
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0 = scores[:, :-1, :-1].argmax(2, keepdim=True).squeeze(2)
        indices1 = scores[:, :-1, :-1].argmax(1, keepdim=True).squeeze(1)
        mutual0 = arange_like(indices0, 1)[None] == paddle_gather(indices1, 1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == paddle_gather(indices0, 1, indices1)
        zero = paddle.to_tensor(0).astype(scores.dtype)
        mscores0 = paddle.where(mutual0, max0.exp(), zero)
        mscores1 = paddle.where(mutual1, paddle_gather(mscores0, 1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.match_threshold)
        valid1 = mutual1 & paddle_gather(valid0, 1, indices1)
        indices0 = paddle.where(valid0, indices0, paddle.to_tensor(-1).astype(indices0.dtype))
        indices1 = paddle.where(valid1, indices1, paddle.to_tensor(-1).astype(indices1.dtype))
        return {
            "matches0": indices0,  # use -1 for invalid match
            "matches1": indices1,  # use -1 for invalid match
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }