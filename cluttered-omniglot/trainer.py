import dataset_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import siamese_u_net
import sys
import tensorflow as tf
import threading
import time
import utils

from queue import Queue
from skimage import io
from tqdm import tqdm

# Memory settings
cpu_cores = [7, 8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

slim = tf.contrib.slim


"""UTILITY FUNCTIONS"""
def get_model(model_name, *args):
    """Get the model given its name."""
    if model_name == "siamese-u-net":
        return siamese_u_net.SiameseUNet(*args)
    else:
        print("Unknown model.")
        exit()


def get_training_placeholders(batch_size, im_height, im_width,
                              tar_height, tar_width):
    images = tf.placeholder(
        tf.float32,
        shape=[batch_size, im_height, im_width, 3],
        name='images')
    labels = tf.placeholder(
        tf.int32,
        shape=[batch_size, im_height, im_width, 1],
        name='labels')
    targets = tf.placeholder(
        tf.float32,
        shape=[batch_size, tar_height, tar_width, 3],
        name='targets')
    return images, labels, targets


def calculate_localization_acc(segmentations, labels):
    """Calculate localization accuracy"""
    lcomx, lcomy = center_of_mass(labels)
    comx, comy = center_of_mass(segmentations)
    euclidian_distance = tf.sqrt((lcomx-comx)**2 + (lcomy-comy)**2)
    distance_metric = tf.cast(euclidian_distance < 5, tf.float32)
    return tf.reduce_mean(distance_metric)


def center_of_mass(image):

    #returns the pixel corresponding to the center of mass of the segmentation mask
    #if no pixel is segmented [-1,-1] is returned
    image = tf.cast(image, tf.float32)

    sz = image.get_shape().as_list()
    batch_size = sz[0]
    szx = sz[1]
    szy = sz[2]

    e = 0.00001

    x,y = tf.meshgrid(list(range(0,szx)),list(range(0,szy)))
    x = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0), [batch_size, 1, 1, 1]), tf.float32)
    y = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(y, axis=-1), axis=0), [batch_size, 1, 1, 1]), tf.float32)
    comx = (tf.reduce_sum(x * image, axis=[1,2,3])+e)//(tf.reduce_sum(image, axis=[1,2,3])-e)
    comy = (tf.reduce_sum(y * image, axis=[1,2,3])+e)//(tf.reduce_sum(image, axis=[1,2,3])-e)

    return comx, comy

def get_shift_xy(image):
    comx, comy = center_of_mass(image)
    sz = image.get_shape().as_list()
    szx = sz[1]
    szy = sz[2]
    shiftx = 2*comx/(szx-1) - 1
    shifty = 2*comy/(szy-1) - 1
    shift = tf.stack([shiftx, shifty])

    return shift


"""DATA GENERATION FUNCTIONS"""
def load_train(fold_dir, index, label_type=None, target_type=None):
    path = os.path.join(fold_dir, 'train/')
    ims_train = io.imread(
        os.path.join(path, 'image_{:08d}.png'.format(index)))
    if label_type:
        seg_train = io.imread(
            os.path.join(path, '{}_segmentation_{:08d}.png'.format(label_type, index)))
    else:
        seg_train = io.imread(
            os.path.join(path, 'segmentation_{:08d}.png'.format(index)))
    if target_type:
        tar_train = io.imread(os.path.join(path, '{}_target_{:08d}.png'.format(target_type, index)))
    else:
        tar_train = io.imread(os.path.join(path, 'target_{:08d}.png'.format(index)))

    return ims_train, seg_train, tar_train


def load_val(fold_dir, index, subset, label_type=None, target_type=None):
    if subset == 'train':
        #load val_train data
        path = os.path.join(fold_dir, 'val-train/')
    elif subset == 'eval':
        #load val_eval data
        path = os.path.join(fold_dir, 'val-one-shot')
    else:
        print(subset + ' is not a valid subset')
    ims_val = io.imread(
        os.path.join(path, 'image_{:08d}.png'.format(index)))
    if label_type:
        seg_val = io.imread(
            os.path.join(path, '{}_segmentation_{:08d}.png'.format(label_type, index)))
    else:
        seg_val = io.imread(
            os.path.join(path, 'segmentation_{:08d}.png'.format(index)))
    if target_type:
        tar_val = io.imread(
            os.path.join(path, '{}_target_{:08d}.png'.format(target_type, index)))
    else:
        tar_val = io.imread(
            os.path.join(path, 'target_{:08d}.png'.format(index)))

    return ims_val, seg_val, tar_val


def load_dataset_test(dataset_dir, subset):
    ims_test = []
    seg_test = []
    tar_test = []
    for index in range(10000):
        ims, seg, tar = load_test(dataset_dir, index, subset)
        ims_test.append(ims)
        seg_test.append(seg)
        tar_test.append(tar)
    return np.concatenate(ims_test), np.concatenate(seg_test), np.concatenate(tar_test)


def load_test(fold_dir, index, subset, label_type=None, target_type=None):
    if subset == 'train':
        #load test_train data
        path = os.path.join(fold_dir, 'test-train/')
    elif subset == 'eval':
        #load test_etest data
        path = os.path.join(fold_dir, 'test-one-shot/')
    else:
        print(subset + ' is not a testid subset')

    ims_test = io.imread(
        os.path.join(path, 'image_{:08d}.png'.format(index)))
    if label_type:
        seg_test = io.imread(
            os.path.join(path, '{}_segmentation_{:08d}.png'.format(label_type, index)))
    else:
        seg_test = io.imread(
            os.path.join(path, 'segmentation_{:08d}.png'.format(index)))
    if target_type:
        tar_test = io.imread(
            os.path.join(path, '{}_target_{:08d}.png'.format(target_type, index)))
    else:
        tar_test = io.imread(
            os.path.join(path, 'target_{:08d}.png'.format(index)))

    return ims_test, seg_test, tar_test


def threaded_batch_generator(generator, batch_size, max_queue_len=1):
    queue = Queue(maxsize=max_queue_len)
    sentinel = object()

    def producer():
        # expect three numpy arrays "blocks"
        ims_batch, seg_batch, tar_batch = [], [], []
        index = 0
        for ims, seg, tar in generator:
            ims_batch.append(ims)
            seg_batch.append(np.expand_dims(seg, axis=2))
            tar_batch.append(np.stack([tar, tar, tar], axis=2))
            index += 1
            if index == batch_size:
                queue.put((
                    np.array(ims_batch), np.array(seg_batch), np.array(tar_batch)))
                ims_batch, seg_batch, tar_batch = [], [], []
                index = 0
        queue.put(sentinel)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def block_generator(
        dataset_dir,
        split,
        modifier,
        label_type,
        target_type,
        num_blocks,
        perms,
        step=0):
    """Yields image triples for training/testing purposes.

    Args:
      dataset_dir: The direct folder where images/segs/targets are held.
      split: "test", "train", or "val"
      modifier: control flow parameter for load_X function above
    """
    while True:
        index = perms[step]
        if split == "train":
            im, seg, tar = load_train(dataset_dir, index, label_type, target_type)
        if split == "val":
            im, seg, tar = load_val(dataset_dir, index, modifier, label_type, target_type)
        if split == "test":
            im, seg, tar = load_test(dataset_dir, index, modifier, label_type, target_type)

        # make seg binary
        seg[seg > 0] = 1
        yield im, seg, tar
        step += 1
        if step == num_blocks:
            step = 0


"""METRIC FUNCTIONS"""
def threshold_segmentations(segmentations, threshold=0.3):
    """Return binary segmentations given segmentations."""
    seg_softmax = tf.nn.softmax(segmentations, axis=-1)
    seg = tf.cast(seg_softmax[...,1] > threshold, tf.int32)
    seg = tf.expand_dims(seg, axis=-1)

    return seg

def calculate_IoU(segmentations, labels, threshold=0.3):
    """Calculate IoU (removing blanks)."""
    pred = tf.squeeze(labels, axis=-1)
    seg = tf.squeeze(segmentations, axis=-1)

    # standard IoU calculation
    IoU = tf.reduce_sum(pred * seg, axis=(1, 2)) / (
        tf.reduce_sum(pred, axis=(1, 2)) + tf.reduce_sum(seg, axis=(1, 2))
        - tf.reduce_sum(pred * seg, axis=(1, 2)))
    # Remove NaNs which appear when the target does not exist
    clean_IoU = tf.where(tf.is_nan(IoU), tf.ones_like(IoU), IoU)
    return clean_IoU


def calculate_mAP(logdir, IoUs, avg_confs, threshold):
    sorted_IoUs = sorted(zip(IoUs, avg_confs), key=lambda p: p[1])
    pos_count = len([item for item in sorted_IoUs if item[0] > threshold])
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for i in range(len(sorted_IoUs)):
        IoU = sorted_IoUs[i][0]
        if IoU > threshold:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / pos_count)
    right_maxes = [i for i in precisions]
    for i in range(len(precisions) - 2, -1, -1):
        right_maxes[i] = max(precisions[i], right_maxes[i + 1])
    width = (recalls[-1] - recalls[0]) / (len(sorted_IoUs) - 1)
    print("AP:", sum(right_maxes) * width)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0.0, 1.0])
    plt.title("PR Curve for threshold={}".format(threshold))
    plt.plot(recalls, right_maxes, c='r')
    plt.plot(recalls, precisions, c='b')
    plt.savefig(os.path.join(logdir, "ap_graph_@_{}.png".format(threshold)))


"""MAIN FUNCTIONS"""
def training(dataset_dir,
             log_dir,
             epochs,
             train_size,
             val_size,
             label_type,
             target_type,
             model_name='siamese-u-net',
             feature_maps=24,
             batch_size=250,
             learning_rate=0.0005,
             dropout=0.0,
             regularization_factor=0.0,
             pretraining_checkpoint=None,
             maximum_number_of_steps=0,
             real_im_path=None):
    model = get_model(model_name, "train", regularization_factor, dropout)
    print("Drawing from {}".format(dataset_dir))

    #Shuffle samples
    perms = np.random.permutation(train_size)
    val_perms = np.random.permutation(val_size)
    if real_im_path:
        real_perms = np.random.permutation(6000)

    with tf.Graph().as_default():

        # Define logging parameters
        t = time.time()
        if pretraining_checkpoint is not None:
            PRETRAINING_CKPT_FILE = pretraining_checkpoint + 'Run.ckpt'
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        # Define training parameters
        batch_size = batch_size
        print("Batch size: {}".format(batch_size))
        print("Dropout: {}".format(dropout))
        # max_steps for epoch training
        # num_blocks for loading block data
        max_train_steps = train_size // batch_size
        num_train_blocks = train_size
        max_val_steps = val_size // batch_size
        num_val_blocks = val_size

        if maximum_number_of_steps != 0:
            print('Going to run for %.d steps' %
                  (np.min([epochs * max_train_steps, maximum_number_of_steps])))
        else:
            print('Going to run for %.d steps' %
                  (epochs * max_train_steps))

        # Get dataset information and statistics
        # Generate batch for this purpose
        # Sample mean from a large number of points
        mean, std = 0.0, 0.0
        sample_im_batch = []
        sample_generator = block_generator(dataset_dir,
                                           split="train",
                                           modifier=False,
                                           label_type=label_type,
                                           target_type=target_type,
                                           num_blocks=1000,
                                           perms=perms)
        # make this the average percentage of the segmentation compared to the whole screen
        active_frac = 0.0
        for i in range(100):
            sample = next(sample_generator)
            im_block, seg, tar_block = sample
            sample_im_batch.append(im_block)
            active_frac += np.sum(seg) / seg.shape[0] / seg.shape[1]
        active_frac /= 100
        print("Found active_frac to be {}.".format(active_frac))

        # mean = np.mean(np.array(sample_im_batch))
        # std = np.std(np.array(sample_im_batch))

        # generate tensorflow placeholders and variables
        images, labels, targets = get_training_placeholders(
            batch_size, im_block.shape[0], im_block.shape[1],
            tar_block.shape[0], tar_block.shape[1]
        )
        learn_rate = tf.Variable(learning_rate)

        # preprocess images
        # targets = (targets - mean)/std
        # images = (images - mean)/std

        # call desired network to get segmentations
        segmentations = model.generate_segmentations(
            targets, images, feature_maps=feature_maps)
        loss_labels = tf.one_hot(tf.squeeze(labels), depth=2)

        # update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        main_loss = tf.losses.compute_weighted_loss(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=loss_labels,
                logits=segmentations,
                pos_weight=1.0/active_frac))
        reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        loss = main_loss + reg_loss

        #Training step
        optimizer = tf.train.AdamOptimizer(learn_rate)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # compute binary segmentations and metrics
        binary_segmentations = threshold_segmentations(segmentations)
        # SR CODE
        tf.reduce_mean(segmentations, axis=[1, 2])
        batch_IoU = calculate_IoU(binary_segmentations, labels)
        count_IoU = tf.count_nonzero(batch_IoU)
        mean_IoU = tf.reduce_mean(batch_IoU)

        # create summaries
        tf.summary.scalar('main_loss', main_loss)
        tf.summary.scalar('regularization_loss', reg_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('mean_IoU', mean_IoU)
        tf.summary.scalar('count_IoU', count_IoU)

        # collect summaries
        summary = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=None)
        restorer = tf.train.Saver(slim.get_model_variables())

        # Initalize batch generators
        train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="train",
                modifier=False,
                label_type=label_type,
                target_type=target_type,
                num_blocks=num_train_blocks,
                perms=perms), batch_size, max_queue_len=100)
        val_train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="val",
                modifier="train",
                label_type=label_type,
                target_type=target_type,
                num_blocks=num_val_blocks,
                perms=val_perms), batch_size)
        val_eval_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="val",
                modifier="eval",
                label_type=label_type,
                target_type=target_type,
                num_blocks=num_val_blocks,
                perms=val_perms), batch_size)

        if real_im_path:
            real_eval_generator = threaded_batch_generator(
                block_generator(
                    real_im_path,
                    split="test",
                    modifier="eval",
                    label_type=label_type,
                    target_type=target_type,
                    num_blocks=6000,
                    perms=real_perms), batch_size)

        print("Beginning session")

        #Start Session
        with tf.Session() as sess:
            #Initialize logging files
            summary_writer_train = tf.summary.FileWriter(log_dir, sess.graph)
            summary_writer_val_train = tf.summary.FileWriter(log_dir + 'val_train')
            summary_writer_val_eval = tf.summary.FileWriter(log_dir + 'val_eval')

            # real image writer, can delete later
            if real_im_path:
                summary_writer_real_eval = tf.summary.FileWriter(log_dir + 'real_eval')


            # Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            if pretraining_checkpoint is not None:
                restorer.restore(sess, PRETRAINING_CKPT_FILE)
                print("Restoring from {}".format(PRETRAINING_CKPT_FILE))

            # Training loop
            print('Starting to train')
            duration = []
            losses = []
            for epoch in range(epochs):
                print("Epoch {}/{}".format(epoch + 1, epochs))
                print("Training for {} steps".format(max_train_steps))

                # learning rate schedule
                if epoch % 2 == 1:
                    learning_rate = learning_rate / 2
                    print('lowering learning rate to %.4f'%(learning_rate))

                losses = []
                last_time = time.time()
                for step in range(max_train_steps):
                    images_batch, labels_batch, target_batch = next(train_generator)
                    _, loss_value, rl = sess.run([train_op, loss, reg_loss],
                                             feed_dict = {targets: target_batch,
                                                          images: images_batch,
                                                          labels: labels_batch,
                                                          learn_rate: learning_rate})
                    new_time = time.time()
                    duration.append(new_time - last_time)
                    last_time = new_time
                    # if step % 5 == 0:
                    #     print("Regularization loss:", rl)
                    #     print("Total loss:", loss_value)

                    # evaluate
                    if step % 200 == 0:
                        #Evaluate and print training error and IoU
                        summary_str_train, tf_IoU = sess.run([summary, mean_IoU],
                                                             feed_dict = {targets: target_batch,
                                                                          images: images_batch,
                                                                          labels: labels_batch})
                        summary_writer_train.add_summary(summary_str_train, step)
                        summary_writer_train.flush()
                        print('Step %5d: loss = %.4f mIoU: %.3f (%.3f sec)'
                              % (step, np.mean(loss_value), tf_IoU, np.mean(duration)))
                        duration = []
                        losses.append(np.mean(loss_value))

                        #evaluate val_train
                        images_batch, labels_batch, target_batch = next(val_train_generator)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch,
                                                                   images: images_batch,
                                                                   labels: labels_batch})
                        summary_writer_val_train.add_summary(summary_str, step)
                        summary_writer_val_train.flush()

                        #evaluate val_eval
                        images_batch, labels_batch, target_batch = next(val_eval_generator)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch,
                                                                   images: images_batch,
                                                                   labels: labels_batch})

                        summary_writer_val_eval.add_summary(summary_str, step)
                        summary_writer_val_eval.flush()

                        if real_im_path:
                            #evaluate real images, can delete later
                            images_batch, labels_batch, target_batch = next(real_eval_generator)
                            summary_str = sess.run(summary, feed_dict={targets: target_batch,
                                                                       images: images_batch,
                                                                       labels: labels_batch})

                            summary_writer_real_eval.add_summary(summary_str, step)
                            summary_writer_real_eval.flush()

                    #Create checkpoint
                    if step % 5000 == 0:
                        checkpoint_file = os.path.join(
                            log_dir, 'Run_Epoch{}_Step{}.ckpt'.format(epoch, step))
                        saver.save(sess, checkpoint_file)
                        checkpoint_file = os.path.join(log_dir, 'Run.ckpt')
                        saver.save(sess, checkpoint_file)


# network evaluation
def evaluation(dataset_dir,
               log_dir,
               test_size,
               label_type,
               target_type,
               model_name='siamese-u-net',
               feature_maps=24,
               batch_size=250,
               dropout=0.0,
               threshold=0.3,
               max_steps=0,
               vis=False):

    model = get_model(model_name, "eval", 0.0, dropout)
    with tf.Graph().as_default():
        # define logging parameters
        EVAL_CKPT_FILE = log_dir + 'Run_Epoch11_Step0.ckpt'# 'Run_Epoch11_Step0.ckpt'
        perms = np.random.permutation(test_size)

        # define training parameters
        batch_size = batch_size
        max_steps = test_size // batch_size
        num_blocks = test_size

        # Get dataset information and statistics
        # Generate batch for this purpose
        # Sample mean from at most 1000 data points
        sample_im_batch = []
        sample_generator = block_generator(dataset_dir,
                                           split="test",
                                           modifier="train",
                                           label_type=label_type,
                                           target_type=target_type,
                                           num_blocks=100,
                                           perms=perms)
        for i in range(100):
            sample = next(sample_generator)
            im_block, _, tar_block = sample
            sample_im_batch.append(im_block)

        # mean = np.mean(np.array(sample_im_batch))
        # std = np.std(np.array(sample_im_batch))

        # generate tensorflow placeholders and variables
        im_block = np.zeros((1, 384, 384, 1))
        tar_block = np.zeros((1, 128, 128, 1))
        images, labels, targets = get_training_placeholders(
            batch_size, im_block.shape[1], im_block.shape[2],
            tar_block.shape[1], tar_block.shape[2]
        )

        # preprocess images
        # targets = (targets - mean)/std
        # images = (images - mean)/std

        # run network and produce segmentations
        segmentations = model.generate_segmentations(
            targets, images, feature_maps=feature_maps)

        # compute binary segmentations and metrics
        binary_segmentations = threshold_segmentations(segmentations)
        mean_IoU = calculate_IoU(binary_segmentations, labels, threshold=threshold)
        mean_dist = calculate_localization_acc(binary_segmentations, labels)

        # logging
        saver = tf.train.Saver()
        restorer = tf.train.Saver(slim.get_model_variables())

        test_train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="test",
                modifier="train",
                label_type=label_type,
                target_type=target_type,
                num_blocks=num_blocks,
                perms=perms), batch_size)
        test_eval_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="test",
                modifier="eval",
                label_type=label_type,
                target_type=target_type,
                num_blocks=num_blocks,
                perms=perms), batch_size)

        # start Session
        with tf.Session() as sess:
            # Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, EVAL_CKPT_FILE)

            # run trainings step
            val_IoU = [0 for x in range(max_steps)]
            os_IoU = [0 for x in range(max_steps)]
            val_distances = [0 for x in range(max_steps)]
            os_distances = [0 for x in range(max_steps)]
            percents = []
            IoUs = []
            avg_confs = []
            for step in tqdm(range(max_steps)):
                # fetch batch
                images_batch, labels_batch, target_batch = next(test_train_generator)

                # perform forward pass and get metrics/segmentations
                val_IoU[step], val_distances[step], segs, confs, ims = sess.run(
                    [mean_IoU, mean_dist, binary_segmentations, segmentations, images],
                    feed_dict = {targets: target_batch,
                                 images: images_batch,
                                 labels: labels_batch})
                # extract confidences and IoUs
                for seg_idx in range(labels_batch.shape[0]):
                    seg = labels_batch[seg_idx]
                    IoUs.append(val_IoU[step][seg_idx])
                    conf = confs[seg_idx]
                    nonzero_conf = conf[np.nonzero(conf)]
                    avg_confs.append(np.sum(nonzero_conf) / nonzero_conf.shape[0])

                # generate confidence heatmap, ground-truth overlay, and predicted mask images
                if vis and step < 25:
                    # select index and fetch all relevant images from batch
                    print(np.unique(ims))
                    saved_idx = np.random.choice(batch_size)
                    im = ims[saved_idx] / 255
                    seg = labels_batch[saved_idx]
                    conf = confs[saved_idx]
                    pred = segs[saved_idx]

                    # set dir
                    vis_dir = os.path.join(log_dir, "vis/")
                    dataset_utils.mkdir_if_missing(vis_dir)

                    # plot and save images
                    plt.figure(figsize=(15, 15))

                    plt.imshow(im)
                    plt.imshow(np.ma.masked_where(seg[...,0] == 0, seg[...,0]), cmap="summer")
                    plt.axis('off')
                    plt.savefig(
                        os.path.join(vis_dir, "gt_overlay_{}.png").format(step),
                        bbox_inches='tight'
                    )
                    plt.clf()

                    plt.imshow(im)
                    plt.imshow(np.ma.masked_where(pred[...,0] == 0, pred[...,0]), cmap="coolwarm")
                    plt.axis('off')
                    plt.savefig(os.path.join(vis_dir, "pred_overlay_{}.png").format(step),
                                bbox_inches='tight')
                    plt.clf()

                    plt.imshow(im)
                    # softmax confs for heatmap
                    conf = utils.softmax(conf, axis=-1)
                    plt.imshow(conf[...,1], cmap="hot")
                    plt.axis('off')
                    plt.savefig(os.path.join(vis_dir, "conf_overlay_{}.png").format(step),
                                bbox_inches='tight')
                    plt.clf()

                    np.save(
                        os.path.join(vis_dir, "sample_pred_{}.npy").format(step),
                        pred,
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_gt_{}.npy").format(step),
                        seg,
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_conf_{}.npy").format(step),
                        conf,
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_tar_{}.npy").format(step),
                        target_batch[saved_idx],
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_im_{}.npy").format(step),
                        im,
                        allow_pickle=False)
                    # normalized images
                    np.save(
                        os.path.join(vis_dir, "norm_im_{}.npy").format(step),
                        images_batch[saved_idx],
                        allow_pickle=False)


                images_batch, labels_batch, target_batch = next(test_eval_generator)
                os_IoU[step], os_distances[step] = sess.run([mean_IoU, mean_dist],
                                         feed_dict = {targets: target_batch,
                                                      images: images_batch,
                                                      labels: labels_batch})

            print('Validation IoU: %.3f'%(np.mean(val_IoU)))
            print('Validation Distance: %.3f'%(np.mean(val_distances)))
            print('One-Shot IoU: %.3f'%(np.mean(os_IoU)), 'One-Shot Distance: %.3f'%(np.mean(os_distances)))

            # use IoUs to compute mAP.
            for t in range(50, 100, 5):
                calculate_mAP(log_dir, IoUs, avg_confs, t / 100)
