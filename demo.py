import cv2
from models import SuperPointFrontend


if __name__ == "__main__":
    weights_path = "models/weights/superpoint_v1.pdparams"

    fe = SuperPointFrontend(weights_path=weights_path,
                            nms_dist=4,
                            conf_thresh=0.015,
                            nn_thresh=0.7)

    im1_path = "assets/images/A.jpg"
    im1 = cv2.imread(im1_path)
    im1_input = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    pts1, desc1, heatmap1 = fe.run(im1_input)
    # print(pts1.shape, desc1.shape, heatmap1.shape)
    cv2.imshow("im1", im1)
    cv2.imshow("heatmap1", heatmap1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()