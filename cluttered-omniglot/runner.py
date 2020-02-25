import trainer
import os

TRAINING = False
USING_SIM = False

if USING_SIM:
    model_name = 'siamese-u-net'

    # locations of data and log, including subfolder/iteration
    DATASET_DIR = os.path.join(
        "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
        "mask-net"
    )

    # WEIGHTS_FOLDER = "sun_fold11_rot1_reg0_drop0"
    # LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + WEIGHTS_FOLDER + '/')
    LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
    FOLD_NUM = 13
    REG_FACTOR = 0
    DROPOUT = 0.5
    BATCH_SIZE = 10
    EVAL_SIZE = 640

    # option for evaluating real images while training sim
    REAL_IM_PATH = "/nfs/diskstation/projects/dex-net/segmentation/datasets/mask-net-real/fold_0010/"

    # Option for dataset maximum sizes
    # lower-bound single rotation
    TRAIN_SIZE = 800000
    VAL_SIZE = 50000
    TEST_SIZE = 50000

    # whether to output/save small # of images for viewing
    VISUALIZE = True

    # training parameters
    LABEL_TYPE = "modal"
    TARGET_TYPE = "amodal"
    PRETRAINING_CKPT_FILE = False
else:
    DATASET_DIR = os.path.join(
        "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
        "mask-net-real"
    )
    FOLD_NUM = 10
    TRAIN_SIZE = 0
    VAL_SIZE = 0
    TEST_SIZE = 6000
    BATCH_SIZE = 1
    BLOCK_SIZE = 1
    WEIGHTS_FOLDER = "sun_fold13_rot4_reg0_drop0"
    LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + WEIGHTS_FOLDER + '/')
    # LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
    VISUALIZE = True
    LABEL_TYPE = "modal"
    TARGET_TYPE = "amodal"


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
                     eval_size=EVAL_SIZE,
                     learning_rate=0.0001,
                     dropout=DROPOUT,
                     regularization_factor=REG_FACTOR,
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
                       dropout=DROPOUT,
                       threshold=0.3,
                       max_steps=0,
                       vis=VISUALIZE)

if __name__ == "__main__":
    if TRAINING:
        train()
    else:
        evaluate()
