import os
import cv2
import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from dataset_utils import mkdir_if_missing
from PIL import Image
from skimage import io
from tqdm import tqdm
import pprint
import json

cpu_cores = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

# input directories
AMODAL_MASK_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/dataset"
SCENE_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/depth_ims"
JSON_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/color_ims"
MASK_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/modal_segmasks"

# output directories
FOLD_NUM = 10
OUT_DIR = "/nfs/diskstation/projects/dex-net/segmentation/datasets/mask-net-real/fold_{:04d}".format(FOLD_NUM)
mkdir_if_missing(OUT_DIR)
mkdir_if_missing(os.path.join(OUT_DIR, "train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-one-shot"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-one-shot"))

# real dataset parameters
NUM_IMS = 400

# original image shape
IM_WIDTH = 1032
IM_HEIGHT = 772

# 1:3 ratio between im_size and tar_size
IM_SIZE = 384
TAR_SIZE = 128

# Image distortion
ANGLE = 180
SHEAR = 0


def rot_x(phi, theta, ptx, pty):
    return np.cos(phi+theta)*ptx + np.sin(phi-theta)*pty


def rot_y(phi, theta, ptx, pty):
    return -np.sin(phi+theta)*ptx + np.cos(phi-theta)*pty


def prepare_img(img, angle=100, shear=2.5, scale=2):
    # Apply affine transformations and scale characters for data augmentation
    phi = np.radians(np.random.uniform(-angle, angle))
    theta = np.radians(np.random.uniform(-shear, shear))
    (x, y) = img.shape
    if scale:
        a = scale**np.random.uniform(-1, 1)
        b = scale**np.random.uniform(-1, 1)
        x = a * x
        y = b * y
    xextremes = [rot_x(phi, theta, 0, 0), rot_x(phi, theta, 0, y), rot_x(phi, theta, x, 0), rot_x(phi, theta, x, y)]
    yextremes = [rot_y(phi, theta, 0, 0), rot_y(phi, theta, 0, y), rot_y(phi, theta, x, 0), rot_y(phi, theta, x, y)]
    mnx = min(xextremes)
    mxx = max(xextremes)
    mny = min(yextremes)
    mxy = max(yextremes)

    aff_bas = np.array([[a*np.cos(phi+theta), b*np.sin(phi-theta), -mnx], [-a*np.sin(phi+theta), b*np.cos(phi-theta), -mny], [0, 0, 1]])
    aff_prm = np.linalg.inv(aff_bas)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.transform((int(mxx - mnx),int(mxy - mny)),
                                    method=Image.AFFINE,
                                    data=np.ndarray.flatten(aff_prm[0:2, :]))
    pil_img = pil_img.resize((int(TAR_SIZE * (mxx - mnx) / 100), int(TAR_SIZE * (mxy - mny) / 100)))

    return np.array(pil_img)


def bbox(im):
    # get bounding box coordinates
    rows = np.any(im, axis=1)
    cols = np.any(im, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def make_target(modal_mask, angle=0, shear=0, scale=1):
    # make target image by cropping
    # formula: use the bigger bounding box length plus half of the smaller
    # margin between the edge of the image and the bbox
    transformed_mask = prepare_img(modal_mask, angle, shear, scale)
    # transformed_mask = modal_mask
    top, bot, left, right = bbox(transformed_mask)
    if bot - top > right - left:
        right += (bot - top - (right - left)) // 2
        left -= (bot - top - (right - left)) // 2
    else:
        bot += (right - left - (bot - top)) // 2
        top -= (right - left - (bot - top)) // 2
    margin = min(top, left, transformed_mask.shape[0] - bot, transformed_mask.shape[1] - right)

    target = cv2.resize(
        transformed_mask[max(0, top - margin):min(transformed_mask.shape[0], bot + margin),
                         max(0, left - margin):min(transformed_mask.shape[1], right + margin)],
        (TAR_SIZE, TAR_SIZE),
        interpolation=cv2.INTER_NEAREST)
    # in the case we have a very zoomed in object, fix that
    if margin < 20:
        padded_target = np.zeros((target.shape[0] + 40, target.shape[1] + 40))
        padded_target[20:padded_target.shape[0] - 20, 20:padded_target.shape[1] - 20] = target
        return cv2.resize(padded_target, (TAR_SIZE, TAR_SIZE), interpolation=cv2.INTER_NEAREST)
    return target


def resize_scene(im):
    if len(im.shape) == 2:
        im = np.pad(im, (((IM_WIDTH - IM_HEIGHT) // 2, (IM_WIDTH - IM_HEIGHT) // 2), (0, 0)), mode="constant")
    elif len(im.shape) == 3:
        im = np.pad(im, (((IM_WIDTH - IM_HEIGHT) // 2, (IM_WIDTH - IM_HEIGHT) // 2), (0, 0), (0, 0)), mode="constant")
    else:
        raise Exception("image dimensions not valid for scene/ground truth, shape: {}".format(im.shape))
    return cv2.resize(
        im,
        (IM_SIZE, IM_SIZE),
        interpolation=cv2.INTER_NEAREST)


# Looping through all image indices
#   Looping through all labels in the json list
#     Get the name
#     Get the target file corresponding to the name
#     Add the image to the batch
#     Process the segmask from modal_segmasks

# Looping through all image indices
#   Looping through all labels in the json list
#     Get the name
#     Get the target file corresponding to the name
#     Add the image to the batch
#     Process the segmask from modal_segmasks

# Looping through all image indices
#   Looping through all labels in the json list
#     Get the name
#     Get the target file corresponding to the name
#     Add the image to the batch
#     Process the segmask from modal_segmasks

data_count = 0
for meta_idx in tqdm(range(NUM_IMS * 30)):
    idx = meta_idx % NUM_IMS
    f = open(os.path.join(JSON_DIR, "image_{:06d}.json".format(idx)))
    json_file = json.load(f)
    scene_path = os.path.join(SCENE_DIR, "image_{:06d}.png".format(idx))
    mask_path = os.path.join(MASK_DIR, "image_{:06d}.png".format(idx))

    # read and blockify scene
    scene_im = io.imread(scene_path)
    scene_im = resize_scene(scene_im)

    # for modal masks
    joint_mask = io.imread(mask_path)
    for label in json_file["labels"]:
        target_name = label["label_class"]
        target_id = label["object_id"]
        target_path = os.path.join(AMODAL_MASK_DIR,
                                   "image_{:06d}".format(idx),
                                   "{}.png".format(target_name))
        try:
            target_im = io.imread(target_path)
            # get first (basically equivalent slice)
            target_im = np.sum(target_im, axis=2)
            modal_target = make_target(target_im, ANGLE, SHEAR)
            modal_target[modal_target > 0] = 1
        except:
            continue

        modal_mask = np.copy(joint_mask)
        modal_mask[modal_mask != target_id] = 0
        modal_mask = resize_scene(modal_mask)
        modal_mask[modal_mask > 0] = 1
        if 1 not in modal_mask:
            continue
        data_count += 1

        # really, all these masks are amodal
        amodal_target = modal_target

        """plt.imshow(scene_im)
        plt.show()
        plt.imshow(modal_mask)
        plt.show()
        plt.imshow(modal_target)
        plt.show()"""

        """print(modal_mask.shape)
        print(amodal_mask.shape)
        print(scene_im.shape)
        print(np.unique(modal_mask))
        print(np.unique(amodal_mask))
        print(np.unique(scene_im))
        print(modal_mask.dtype)
        print(amodal_mask.dtype)
        print(scene_im.dtype)"""

        io.imsave(os.path.join(
            OUT_DIR,
            "test-train/",
            "image_{:08d}.png".format(meta_idx)),
               scene_im)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-train/",
            "modal_segmentation_{:08d}.png".format(meta_idx)),
               modal_mask)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-train/",
            "modal_target_{:08d}.png".format(meta_idx)),
               modal_target)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-train/",
            "amodal_target_{:08d}.png".format(meta_idx)),
               amodal_target)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "image_{:08d}.png".format(meta_idx)),
               scene_im)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "modal_segmentation_{:08d}.png".format(meta_idx)),
               modal_mask)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "modal_target_{:08d}.png".format(meta_idx)),
               modal_target)
        io.imsave(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "amodal_target_{:08d}.png".format(meta_idx)),
               amodal_target)
