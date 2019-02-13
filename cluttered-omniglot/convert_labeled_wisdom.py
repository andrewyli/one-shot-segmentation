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

# input directories
AMODAL_MASK_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/dataset"
SCENE_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/depth_ims"
JSON_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/color_ims"
MASK_DIR = "/nfs/diskstation/dmwang/labeled_wisdom_real/phoxi/modal_segmasks"

# output directories
OUT_DIR = "/nfs/diskstation/projects/dex-net/segmentation/datasets/mask-net-real/fold_0000"
mkdir_if_missing(OUT_DIR)
mkdir_if_missing(os.path.join(OUT_DIR, "train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-one-shot"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-one-shot"))

# real dataset parameters
NUM_IMS = 400

# original image shape
IM_WIDTH = 772
IM_HEIGHT = 1032


# 1:3 ratio between im_size and tar_size
IM_SIZE = 384
TAR_SIZE = 128

# Image distortion
ANGLE = 100
SHEAR = 4

# For storage purposes
BLOCK_SIZE = 500

def rot_x(phi, theta, ptx, pty):
    return np.cos(phi+theta)*ptx + np.sin(phi-theta)*pty


def rot_y(phi, theta, ptx, pty):
    return -np.sin(phi+theta)*ptx + np.cos(phi-theta)*pty


def prepare_img(img, angle=100, shear=2.5, scale=2):
    # Apply affine transformations and scale characters for data augmentation
    phi = np.radians(np.random.uniform(-angle, angle))
    theta = np.radians(np.random.uniform(-shear, shear))
    a = scale**np.random.uniform(-1, 1)
    b = scale**np.random.uniform(-1, 1)
    (x, y) = img.shape
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
    # make target image
    transformed_mask = prepare_img(modal_mask, angle, shear, scale)
    top, bot, left, right = bbox(transformed_mask)
    obj_size = max(bot - top, right - left)
    margin = max((TAR_SIZE * 2 - obj_size) // 2, 0)
    return cv2.resize(
        transformed_mask[max(0, top - margin):min(transformed_mask.shape[0], bot + margin),
                         max(0, left - margin):min(transformed_mask.shape[1], right + margin)],
        (TAR_SIZE, TAR_SIZE),
        interpolation=cv2.INTER_NEAREST)

def resize_scene(im):
    if len(im.shape) == 2:
        im = np.pad(im, (((IM_WIDTH - IM_SIZE) // 2, (IM_WIDTH - IM_SIZE) // 2), (0, 0)), mode="constant")
    elif len(im.shape) == 3:
        im = np.pad(im, (((IM_WIDTH - IM_SIZE) // 2, (IM_WIDTH - IM_SIZE) // 2), (0, 0), (0, 0)), mode="constant")
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

data_count = 0
for meta_idx in tqdm(range(NUM_IMS * 30)):
    idx = meta_idx % NUM_IMS
    f = open(os.path.join(JSON_DIR, "image_{:06d}.json".format(idx)))
    json_file = json.load(f)
    scene_path = os.path.join(SCENE_DIR, "image_{:06d}.png".format(idx))
    mask_path = os.path.join(MASK_DIR, "image_{:06d}.png".format(idx))

    # read and blockify scene
    scene_im = io.imread(scene_path, as_gray=True)
    scene_im = resize_scene(scene_im)

    # triplicate the scene (384, 384) --> (384, 384, 3)
    scene_im = np.stack([scene_im, scene_im, scene_im], axis=2)

    scene_im = np.expand_dims(scene_im, axis=0)

    # for modal masks
    joint_mask = io.imread(mask_path, as_gray=True)
    for label in json_file["labels"]:
        target_name = label["label_class"]
        target_id = label["object_id"]
        target_path = os.path.join(AMODAL_MASK_DIR,
                                   "image_{:06d}".format(idx),
                                   "{}.png".format(target_name))
        try:
            target_im = io.imread(target_path, as_gray=True)
            amodal_mask = make_target(target_im, angle=0, shear=0)
            amodal_mask[amodal_mask > 0] = 1.0
            """print(np.unique(amodal_mask))
            plt.imshow(amodal_mask)
            plt.show()"""
        except:
            continue

        modal_mask = np.copy(joint_mask)
        modal_mask[modal_mask != target_id] = 0
        modal_mask = resize_scene(modal_mask)
        modal_mask[modal_mask > 0] = 1.0
        data_count += 1

        """plt.imshow(scene_im[0])
        plt.show()
        plt.imshow(modal_mask)
        plt.show()
        plt.imshow(amodal_mask)
        plt.show()"""

        # triplicate the amodal mask (128, 128) --> (128, 128, 3)
        amodal_mask = np.stack([amodal_mask, amodal_mask, amodal_mask], axis=2)

        # create 1-sized blocks
        amodal_mask = np.expand_dims(amodal_mask, axis=0)
        modal_mask = np.expand_dims(modal_mask, axis=2)
        modal_mask = np.expand_dims(modal_mask, axis=0)

        """print(modal_mask.shape)
        print(amodal_mask.shape)
        print(scene_im.shape)
        print(np.unique(modal_mask))
        print(np.unique(amodal_mask))
        print(np.unique(scene_im))
        print(modal_mask.dtype)
        print(amodal_mask.dtype)
        print(scene_im.dtype)"""

        np.save(os.path.join(
            OUT_DIR,
            "test-train/",
            "image_{:08d}.npy".format(meta_idx)),
               scene_im)
        np.save(os.path.join(
            OUT_DIR,
            "test-train/",
            "segmentation_{:08d}.npy".format(meta_idx)),
               modal_mask)
        np.save(os.path.join(
            OUT_DIR,
            "test-train/",
            "target_{:08d}.npy".format(meta_idx)),
               amodal_mask)
        np.save(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "image_{:08d}.npy".format(meta_idx)),
               scene_im)
        np.save(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "segmentation_{:08d}.npy".format(meta_idx)),
               modal_mask)
        np.save(os.path.join(
            OUT_DIR,
            "test-one-shot/",
            "target_{:08d}.npy".format(meta_idx)),
               amodal_mask)
