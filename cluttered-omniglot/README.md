# README

First you will need to generate the data, then you will be able to train on it.

## Generating data:

To generate the simulation data from Wisdom, and convert it into an appropriate form for training this model, run `convert_table_clutter.py`.
Download Wisdom (https://sites.google.com/view/wisdom-dataset/dataset_links)
Set `DATASET_DIR` to the location of Wisdom and `OUT_DIR` to a new folder
Run the file with `python convert_table_clutter.py`
*Note*: The notebook was only useful for testing the correctness of this code and looking at a small number of samples - due to the full size of this dataset it is not advised to use the notebook to do the heavy lifting. Use the python script instead.


## Running the training code:

To train the model, use the training file `runner.py`.
Keep the sizes of the dataset the same as they are.
Set the dataset directory to the location of your generated data, and make a log directory where you want to store your checkpoint and visualization files. Set that directory under `LOGDIR`
Set the `BATCH_SIZE` to whatever is viable given the space on your GPU/TPU.
Keep `VISUALIZE` True if you want to save the first 25 images to a new folder (`vis/`) within your `logdir`.
Set the `LABEL_TYPE` and `TARGET_TYPE` to do amodal --.> modal segmentation for example, or whichever mode you want.
Run this with python, `CUDA_VISIBLE_DEVICES=[whatever device you're using, 0, 1, 2 etc] python runner.py`


## File overview
### `runner.py`
This file is the runner for all training and evaluation. At the top are the parameters for model type, dataset, logging, batch_size, etc.

### `trainer.py`
This is the file that contains the training and evalution methods to be run by `runner.py`. It has the main training and evaluation loops at the bottom, the data batch generator functions at the top, and metrics calculating functions in the middle.

### `siamese_u_net.py`
This file defines the siamese u-net architecture, and is called by `trainer.py` when producing segmentations.
