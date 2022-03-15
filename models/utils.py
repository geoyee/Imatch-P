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


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text=[], path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[: H0, : W0, :] = image0
    out[: H1, W0 + margin:, :] = image1
    # out = np.stack([out] * 3, -1)
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (242, 242, 242)
        red = (55, 57, 207)  # BGR
        for x, y in kpts0[0]:
            cv2.circle(out, (x, y), 2, white, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, red, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1[0]:
            cv2.circle(out, (x + margin + W0, y), 2, white, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, red, -1,
                       lineType=cv2.LINE_AA)
    # mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    # color = (np.array(color[:, : 3]) * 255).astype(int)[:, ::-1]
    # for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
    #     c = c.tolist()
    #     cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
    #              color=c, thickness=1, lineType=cv2.LINE_AA)
    #     # display line end-points as circles
    #     cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
    #     cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
    #                lineType=cv2.LINE_AA)
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)
    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)
    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out