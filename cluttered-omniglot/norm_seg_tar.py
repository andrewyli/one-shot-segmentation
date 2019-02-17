import numpy as np
import os
from tqdm import tqdm


# read params
DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    "mask-net"
)
FOLD_NUM = 2
SET_NAME = "train"
SET_SIZE = 258700
BLOCK_SIZE = 50

num_blocks = SET_SIZE // BLOCK_SIZE
base_path = os.path.join(DATASET_DIR, "fold_{:04d}".format(FOLD_NUM), SET_NAME)

# get path function
get_path = lambda name, idx: os.path.join(base_path, "{}_{:08d}.npy".format(name, idx))

for block_idx in tqdm(range(num_blocks)):
    try:
        seg_block = np.load(get_path("segmentation", block_idx))
        tar_block = np.load(get_path("target", block_idx))
    except:
        break

    seg_block = seg_block.astype("uint8")
    tar_block = tar_block.astype("uint8")

    seg_block[seg_block == 255] = 1
    tar_block[tar_block == 255] = 1

    np.save(get_path("segmentation", block_idx), seg_block)
    np.save(get_path("target", block_idx), tar_block)
