from models import Matching, read_image, make_matching_plot_fast


if __name__ == "__main__":
    model = Matching()
    im0, imt0 = read_image("assets/images/DSC03193.JPG")
    im1, imt1 = read_image("assets/images/DSC03194.JPG")
    data = {"image0": imt0, "image1": imt1}
    pred = model(data)
    # print(pred)
    make_matching_plot_fast(
        im0, im1, pred["keypoints0"], pred["keypoints1"], 
        pred["matches0"], pred["matches1"])