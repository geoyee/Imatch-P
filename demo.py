import matplotlib.cm as cm
from models import Matching, read_image, make_matching_plot_fast


if __name__ == "__main__":
    model = Matching()
    im0, imt0 = read_image("assets/images/A.jpg", (128, 128))
    im1, imt1 = read_image("assets/images/B.jpg", (128, 128))
    data = {"image0": imt0, "image1": imt1}
    pred = model(data)
    # print(pred)
    valid = pred["matches0"] > -1
    make_matching_plot_fast(
        im0, im1, pred["keypoints0"], pred["keypoints1"], 
        pred["keypoints0"], pred["keypoints1"], 
        color=cm.jet, text=["matching"], show_keypoints=True, opencv_display=True
    )