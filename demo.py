from models import Matching, read_image, make_matching_plot_fast


def matching_and_display(im0_path, im1_path):
    model = Matching()
    im0, imt0 = read_image(im0_path)
    im1, imt1 = read_image(im1_path)
    data = {"image0": imt0, "image1": imt1}
    pred = model(data)
    # print(pred)
    out = make_matching_plot_fast(
        im0, im1, pred["keypoints0"], pred["keypoints1"], 
        pred["matches0"], pred["matches1"])


if __name__ == "__main__":
    im0_path = "assets/A.jpg"
    im1_path = "assets/B.jpg"
    matching_and_display(im0_path, im1_path)
