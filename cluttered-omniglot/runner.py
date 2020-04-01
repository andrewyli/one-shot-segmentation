import argparse
import os

from autolab_core import YamlConfig
import trainer


def train(config):
    datadir = os.path.join(
        config['dataset_dir'], 
        "fold_{:04d}".format(config['fold_num']))
    logdir = os.path.join(config['log_dir'], config['model_name'])
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    trainer.training(datadir,
                     logdir,
                     epochs=config['epochs'],
                     train_size=config['train_size'],
                     val_size=config['val_size'],
                     label_type=config['label_type'],
                     target_type=config['target_type'],
                     model_name=config['model_type'],
                     feature_maps=config['feature_maps'],
                     batch_size=config['batch_size'],
                     eval_size=config['eval_size'],
                     learning_rate=config['lr'],
                     dropout=config['dropout'],
                     regularization_factor=config['reg_factor'],
                     pretraining_checkpoint=logdir if config['pretrained'] else None,
                     maximum_number_of_steps=0,
                     real_im_path=config['real_im_path'])

def evaluate(config):
    datadir = os.path.join(
        config['dataset_dir'], "fold_{:04d}".format(config['fold_num']))
    logdir = os.path.join(config['log_dir'], config['model_name'])

    trainer.evaluation(datadir,
                       logdir,
                       test_size=config['test_size'],
                       label_type=config['label_type'],
                       target_type=config['target_type'],
                       model_name=config['model_type'],
                       feature_maps=config['feature_maps'],
                       batch_size=config['batch_size'],
                       dropout=config['dropout'],
                       threshold=0.3,
                       max_steps=0,
                       vis=config['vis'])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='training and evaluation for OSS')
    parser.add_argument('--cfg', type=str, default='cfg/runner.yaml', help='config file with parameters for eval/training')
    args = parser.parse_args()

    config = YamlConfig(args.cfg)

    if config['train']:
        train(config)
    else:
        evaluate(config)
