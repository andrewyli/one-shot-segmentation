# COPY FROM NOTEBOOK
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset_utils import mkdir_if_missing
from PIL import Image
from skimage import io
from tqdm import tqdm

import json
import pprint

# Set false to save images instead of displaying them (Default)
DISPLAY_ONLY = False

# DO NOT COPY FROM NOTEBOOK - OS RELATED
cpu_cores = [0, 1, 2, 3, 4, 5, 6, 7, 8] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

# SET FOLD_NUM
DATASET_DIR = "/nfs/diskstation/projects/dex-net/segmentation/datasets/wisdom-sim-block-npy"
FOLD_NUM = 33
OUT_DIR = "/nfs/diskstation/projects/dex-net/segmentation/datasets/mask-net/fold_{:04d}".format(FOLD_NUM)
print(OUT_DIR)
mkdir_if_missing(OUT_DIR)
mkdir_if_missing(os.path.join(OUT_DIR, "train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "val-one-shot"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-train"))
mkdir_if_missing(os.path.join(OUT_DIR, "test-one-shot"))

# COPY FROM NOTEBOOK BUT KEEP NUM_IMS 50000
# Dataset size (OG: 50000, rotations 1)
NUM_IMS = 12500
NUM_ROTATIONS = 4

# input image size
IM_HEIGHT = 384
IM_WIDTH = 512

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


def prepare_img(img, angle=90, shear=0, scale=None):
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


data_count = 0
print(DISPLAY_ONLY)
train_indices = set(np.load(
    os.path.join(DATASET_DIR, "train_indices.npy")))
test_indices = set(np.load(
    os.path.join(DATASET_DIR, "test_indices.npy")))

train_counter = 0
test_counter = 0
test_folders = ["test-train", "test-one-shot", "val-train", "val-one-shot"]
test_counters = [0 for i in range(4)]
for c_idx in tqdm(range(NUM_IMS)):
    # because Wisdom is grouped in sets of five identical images, this will ensure subsets will keep the same number of unique states
    im_idx = 5 * (c_idx % 10000) + (c_idx // 10000)
    im_path = os.path.join(
        DATASET_DIR,
        "depth_ims",
        "image_{:06d}.png".format(im_idx))
    im = io.imread(im_path)
    im = resize_scene(im)

    for mask_idx in range(11):
        for rot in range(NUM_ROTATIONS):
            channel_name = "image_{:06d}_channel_{:03d}.png".format(im_idx, mask_idx)
            amodal_path = os.path.join(
                DATASET_DIR,
                "amodal_segmasks",
                channel_name)
            amodal_image = io.imread(amodal_path)
            amodal_mask = resize_scene(amodal_image)
            amodal_mask[amodal_mask > 0] = 1

            modal_path = os.path.join(
                DATASET_DIR,
                "modal_segmasks",
                channel_name)
            modal_image = io.imread(modal_path)
            modal_mask = resize_scene(modal_image)
            modal_mask[modal_mask > 0] = 1

            # if the object is completely invisible, omit from the dataset
            if (len(np.unique(modal_mask)) == 1):
                continue

            # if there is no amodal mask, object doesn't exist.
            try:
                amodal_target = make_target(amodal_mask, angle=ANGLE, shear=SHEAR)
                # capture original object
                amodal_model = np.copy(amodal_target)
                amodal_target[amodal_target > 0] = 1

                # in case of modal->modal segmentation
                modal_target = make_target(modal_image, angle=ANGLE, shear=SHEAR)
                modal_target[modal_target > 0] = 1
            except:
                continue

            data_count += 1


            if DISPLAY_ONLY:
                plt.figure()
                plt.title("scene")
                plt.imshow(im)
                plt.figure()
                plt.title("amodal mask")
                plt.imshow(amodal_mask)
                plt.figure()
                plt.title("modal mask")
                plt.imshow(modal_mask)
                plt.figure()
                plt.title("amodal target")
                plt.imshow(amodal_target)
                plt.figure()
                plt.title("modal target")
                plt.imshow(modal_target)
                continue

            if im_idx in train_indices:
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "image_{:08d}.png".format(train_counter)),
                       im)
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "amodal_segmentation_{:08d}.png".format(train_counter)),
                          amodal_mask)
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "modal_segmentation_{:08d}.png".format(train_counter)),
                       modal_mask)
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "amodal_target_{:08d}.png".format(train_counter)),
                          amodal_target)
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "modal_target_{:08d}.png".format(train_counter)),
                          modal_target)
                io.imsave(os.path.join(
                    OUT_DIR,
                    "train/",
                    "amodal_model_{:08d}.png".format(train_counter)),
                          amodal_model)
                train_counter += 1

            elif im_idx in test_indices:
                test_folder_idx = np.random.choice([0, 1, 2, 3])
                test_folder = test_folders[test_folder_idx]
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "image_{:08d}.png".format(test_counters[test_folder_idx])),
                        im)
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "amodal_segmentation_{:08d}.png".format(test_counters[test_folder_idx])),
                          amodal_mask)
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "modal_segmentation_{:08d}.png".format(test_counters[test_folder_idx])),
                          modal_mask)
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "amodal_target_{:08d}.png".format(test_counters[test_folder_idx])),
                          amodal_target)
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "modal_target_{:08d}.png".format(test_counters[test_folder_idx])),
                          modal_target)
                io.imsave(os.path.join(
                    OUT_DIR,
                    test_folder,
                    "amodal_model_{:08d}.png".format(test_counters[test_folder_idx])),
                          amodal_model)
                test_counters[test_folder_idx] += 1
                test_counter += 1
print("Data count for {} images: {}".format(NUM_IMS, data_count))
