import trainer
import os


model_name = 'siamese-u-net'

# locations of data and log, including subfolder/iteration
DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    "mask-net"
)
LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
FOLD_NUM = 9

# option for evaluating real images while training sim
REAL_IM_PATH = "/nfs/diskstation/projects/dex-net/segmentation/datasets/mask-net-real/fold_0002/"

# option for dataset maximum sizes
# TRAIN_SIZE = 214720
# VAL_SIZE = 13279
# TEST_SIZE = 13223
TRAIN_SIZE = 1030779
TEST_SIZE = 63537
VAL_SIZE = 64153

# batch size of training/eval
BATCH_SIZE = 10

# whether to output/save small # of images for viewing
VISUALIZE = True

# training parameters
LABEL_TYPE = "modal"
TARGET_TYPE = "modal"
PRETRAINING_CKPT_FILE = False

# DATASET_DIR = os.path.join(
#     "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
#     "mask-net-real"
# )
# FOLD_NUM = 1
# TRAIN_SIZE = 0
# VAL_SIZE = 0
# TEST_SIZE = 6000
# BATCH_SIZE = 1
# BLOCK_SIZE = 1
# LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
# VISUALIZE = True
# LABEL_TYPE = "modal"
# TARGET_TYPE = "modal"


def train():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR

    trainer.training(datadir,
                     logdir,
                     epochs=20,
                     train_size=TRAIN_SIZE,
                     val_size=VAL_SIZE,
                     label_type=LABEL_TYPE,
                     target_type=TARGET_TYPE,
                     model_name=model_name,
                     feature_maps=24,
                     batch_size=BATCH_SIZE,
                     learning_rate=0.00005,
                     pretraining_checkpoint=logdir if PRETRAINING_CKPT_FILE else None,
                     maximum_number_of_steps=0,
                     real_im_path=REAL_IM_PATH)

def evaluate():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR

    trainer.evaluation(datadir,
                       logdir,
                       test_size=TEST_SIZE,
                       label_type=LABEL_TYPE,
                       target_type=TARGET_TYPE,
                       model_name=model_name,
                       feature_maps=24,
                       batch_size=BATCH_SIZE,
                       threshold=0.3,
                       max_steps=0,
                       vis=VISUALIZE)

if __name__ == "__main__":
    # train()
    evaluate()
