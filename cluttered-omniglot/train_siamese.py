import model
import os

model_name = 'siamese-u-net'
DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    'cluttered_omniglot/'
)
FOLD_NUM = 5
TRAIN_SIZE = 2000000
VAL_SIZE = 10000
TEST_SIZE = 10000
BLOCK_SIZE = 250
LOG_DIR = os.path.join(os.getcwd(), 'logs/' + model_name + '/')

NUM_CHARS = 4

def train():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "{}_characters".format(NUM_CHARS), "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR + '%.d_characters/' % (NUM_CHARS)

    model.training(datadir,
                   logdir,
                   epochs=20,
                   train_size=TRAIN_SIZE,
                   val_size=VAL_SIZE,
                   block_size=BLOCK_SIZE,
                   feature_maps=24,
                   batch_size=250,
                   learning_rate=0.0005,
                   maximum_number_of_steps=0)

def evaluate():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "{}_characters".format(NUM_CHARS), "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR + '%.d_characters/'%(NUM_CHARS)

    model.evaluation(datadir,
                     logdir,
                     test_size=TEST_SIZE,
                     block_size=BLOCK_SIZE,
                     model=model_name,
                     feature_maps=24,
                     batch_size=250,
                     threshold=0.3,
                     max_steps=0)

if __name__ == "__main__":
    train()
    # evaluate()
