import os
import numpy as np
import skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from scipy.misc import imsave

base_path = "images/"
out_path = "histograms/"

images = ["image_00000000.npy"]
boxes_list = [[[650, 650, 730, 730]]]


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def analyze_image_depths(path, bbox, out_name):
    """
    path should lead to a .npy file
    """
    img = np.load(path)
    # since these are triplicated, analyze only one dimension
    img = img[:, :, 0]
    img_slice = img[bbox[0] : bbox[2], bbox[1]: bbox[3]]
    vec = np.ndarray.flatten(img_slice)
    # vec = reject_outliers(vec)

    var = np.var(vec)
    mean = np.mean(vec)
    print("State for {}: Mean: {}, Standard Deviation: {}\n".format(out_name, mean, np.sqrt(var)))

    n, bins, patches = plt.hist(vec, vec.size // 10, facecolor="blue")

    plt.xlabel("depth value")
    plt.ylabel("count")
    plt.title("depth within region")
    plt.grid(True)
    plt.show()

    plt.savefig(os.path.join(out_path, "graph_" + out_name), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for img_name in images:
        img_path = os.path.join(base_path, img_name)
        img = np.load(img_path)
        # since these are triplicated, analyze only one dimension
        img = img[:, :, 0]
        imsave(os.path.join(base_path, "{}.png".format(img_name[:-4])), img)

    for img_name, boxes in zip(images, boxes_list):
        img_path = os.path.join(base_path, img_name)
        for box in boxes:
            out_name = "img_{}_box_{}_{}_{}_{}.png" \
                       .format(img_name, box[0], box[1], box[2], box[3])
            analyze_image_depths(img_path, box, out_name)
