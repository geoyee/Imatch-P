import cv2
import numpy as np
import paddle


def img2tensor(img):
    return paddle.to_tensor(img / 255.).astype("float32")[None, None]


def read_image(path, size=None):
    image = cv2.imread(path)
    if size is not None:
        image = cv2.resize(image, size)
    image_tensor = img2tensor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return image, image_tensor


def make_matching_plot_fast(im0, im1, kpts0, kpts1, matches0, matches1, 
                            margin=10, path=None, show_points=True, display=True):
    H0, W0 = im0.shape[:2]
    H1, W1 = im1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[: H0, : W0, :] = im0
    out[: H1, W0 + margin:, :] = im1
    if show_points:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (242, 242, 242)
        red = (55, 57, 207)  # BGR
        for x, y in kpts0[0]:
            cv2.circle(out, (x, y), 2, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1[0]:
            cv2.circle(out, (x + margin + W0, y), 2, red, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)
    # add lines
    mkpts0 = kpts0[(matches0 > -1).numpy()]
    mkpts1 = kpts1[(matches1 > -1).numpy()]
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        white = (242, 242, 242)
        turquoise = (149 ,166, 57)  # BGR
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=turquoise, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, turquoise, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0), 1, white, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, turquoise, -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 1, white, -1, 
                   lineType=cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
    if display:
        cv2.imshow("matching", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out