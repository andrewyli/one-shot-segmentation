import model
import os

model_name = 'mask-net'
DATASET_DIR = os.path.join(
    "/nfs/diskstation/projects/dex-net/segmentation/datasets/",
    'cluttered_omniglot/'
)
FOLD_NUM = 1
TRAIN_SIZE = 2000000
VAL_SIZE = 10000
TEST_SIZE = 10000
BLOCK_SIZE = 250
LOG_DIR = os.path.join(os.getcwd(), 'logs/' + model_name + '/')

# Train encoder and discriminator
# A trained Siamese-U-Net model is required
NUM_CHARS = 10

def train_encoder():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "{}_characters".format(NUM_CHARS), "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR + 'encoder_decoder/%.d_characters/'%(NUM_CHARS)
    ckptdir = os.path.join(os.getcwd(), 'logs/siamese-u-net/') + '%.d_characters/'%(NUM_CHARS)

    model.training(datadir,
                   logdir,
                   epochs=5,
                   train_size=TRAIN_SIZE,
                   val_size=VAL_SIZE,
                   block_size=BLOCK_SIZE,
                   model=model_name,
                   train_mode='encoder_decoder',
                   feature_maps=24,
                   batch_size=250,
                   learning_rate=0.00005,
                   pretraining_checkpoint=ckptdir,
                   maximum_number_of_steps=0)

def train_discriminator():
    # Train the discriminator
    print('')
    datadir = os.path.join(
        DATASET_DIR, "{}_characters".format(NUM_CHARS), "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR + 'discriminator/%.d_characters/'%(NUM_CHARS)
    ckptdir = LOG_DIR + 'encoder_decoder/%.d_characters/'%(NUM_CHARS)

    model.training(datadir,
                   logdir,
                   epochs=5,
                   test_size=TEST_SIZE,
                   block_size=BLOCK_SIZE,
                   model=model_name,
                   train_mode='discriminator',
                   feature_maps=24,
                   batch_size=250,
                   learning_rate=0.00025,
                   pretraining_checkpoint=ckptdir,
                   maximum_number_of_steps=0)

def evaluate():
    print('')
    datadir = os.path.join(
        DATASET_DIR, "{}_characters".format(NUM_CHARS), "fold_{:04d}".format(FOLD_NUM))
    logdir = LOG_DIR + 'discriminator/%.d_characters/'%(NUM_CHARS)

    model.evaluation(datadir,
                     logdir,
                     model=model_name,
                     feature_maps=24,
                     batch_size=50,
                     threshold=0.3,
                     max_steps=0)

if __name__ == "__main__":
    train_encoder()
    # train_discriminator()
    # evaluate()
