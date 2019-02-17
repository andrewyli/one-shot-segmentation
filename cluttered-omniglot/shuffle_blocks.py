import numpy as np
import os
from tqdm import tqdm


# read params
DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    "mask-net"
)
FOLD_NUM = 0
SET_NAME = "test-one-shot"
SET_SIZE = 258700
BLOCK_SIZE = 50
NUM_SHUFFLE = 14
num_ims = NUM_SHUFFLE * BLOCK_SIZE

# write params
WRITE_FOLD_NUM = 2
write_path = os.path.join(DATASET_DIR, "fold_{:04d}".format(WRITE_FOLD_NUM), SET_NAME)

# constructed params
base_path = os.path.join(DATASET_DIR, "fold_{:04d}".format(FOLD_NUM), SET_NAME)
num_blocks = SET_SIZE // BLOCK_SIZE

# tracking idxs
block_idx = 0
left, right = block_idx, block_idx

# get path function
get_path = lambda name, idx: os.path.join(base_path, "{}_{:08d}.npy".format(name, idx))


# get dimensions
im_shape = np.load(get_path("image", block_idx)).shape
seg_shape = np.load(get_path("segmentation", block_idx)).shape
tar_shape = np.load(get_path("target", block_idx)).shape

# create block pool (replace first dimension with big fatty)
im_pool = np.zeros([num_ims] + list(im_shape)[1:])
seg_pool = np.zeros([num_ims] + list(seg_shape)[1:])
tar_pool = np.zeros([num_ims] + list(tar_shape)[1:])


print("Reading from: {}".format(base_path))
print("Running {} images in set {}.".format(SET_SIZE, SET_NAME))
print("BLOCK_SIZE: {}".format(BLOCK_SIZE))
print("Runing for {} iterations.".format(num_blocks))
print("Number of blocks to shuffle: {}".format(NUM_SHUFFLE))
print("Saving to {}".format(write_path))


pbar = tqdm(total=num_blocks)
while block_idx < num_blocks:
    im_block = np.load(get_path("image", block_idx))
    seg_block = np.load(get_path("segmentation", block_idx))
    tar_block = np.load(get_path("target", block_idx))

    # read in block
    im_pool_idx = right - left
    im_pool[im_pool_idx:im_pool_idx + BLOCK_SIZE, :, :, :] = im_block
    seg_pool[im_pool_idx:im_pool_idx + BLOCK_SIZE, :, :, :] = seg_block
    tar_pool[im_pool_idx:im_pool_idx + BLOCK_SIZE, :, :, :] = tar_block

    right += BLOCK_SIZE
    block_idx += 1

    if right - left >= num_ims:
        perm = np.random.permutation(num_ims)
        for new_block_idx in range(left, right, BLOCK_SIZE):
            new_im_block = im_pool[perm[new_block_idx - left:new_block_idx - left + BLOCK_SIZE]]
            new_seg_block = seg_pool[perm[new_block_idx - left:new_block_idx - left + BLOCK_SIZE]]
            new_tar_block = tar_pool[perm[new_block_idx - left:new_block_idx - left + BLOCK_SIZE]]
            np.save(os.path.join(write_path, "image_{:08d}.npy".format(new_block_idx // BLOCK_SIZE)), new_im_block)
            np.save(os.path.join(write_path, "segmentation_{:08d}.npy".format(new_block_idx // BLOCK_SIZE)), new_seg_block)
            np.save(os.path.join(write_path, "target_{:08d}.npy".format(new_block_idx // BLOCK_SIZE)), new_tar_block)
        pbar.update(NUM_SHUFFLE)
        left = right
pbar.close()
