import model
import os

model_name = 'siamese-u-net'

DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    "mask-net"
)
FOLD_NUM = 1
TRAIN_SIZE = 258700
VAL_SIZE = 1480
TEST_SIZE = 1480
BATCH_SIZE = 10
BLOCK_SIZE = 50
LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
VISUALIZE = False


# DATASET_DIR = os.path.join(
#     "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
#     "mask-net-real"
# )
# FOLD_NUM = 0
# TRAIN_SIZE = 0
# VAL_SIZE = 0
# TEST_SIZE = 6000
# BATCH_SIZE = 1
# BLOCK_SIZE = 1
# LOG_DIR = os.path.join(os.getcwd(), 'logs/' + "clutter/" + model_name + '/')
# VISUALIZE = False


def train():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR

    model.training(datadir,
                   logdir,
                   epochs=20,
                   train_size=TRAIN_SIZE,
                   val_size=VAL_SIZE,
                   block_size=BLOCK_SIZE,
                   feature_maps=24,
                   batch_size=BATCH_SIZE,
                   learning_rate=0.00005,
                   maximum_number_of_steps=0)

def evaluate():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR

    model.evaluation(datadir,
                     logdir,
                     test_size=TEST_SIZE,
                     block_size=BLOCK_SIZE,
                     model=model_name,
                     feature_maps=24,
                     batch_size=BATCH_SIZE,
                     threshold=0.3,
                     max_steps=0,
                     vis=VISUALIZE)

if __name__ == "__main__":
    train()
    # evaluate()
