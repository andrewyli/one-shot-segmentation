import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, deque
import sys

# Memory settings
import os
cpu_cores = [7, 8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

import time

from tqdm import tqdm

slim = tf.contrib.slim

from joblib import Parallel, delayed
from multiprocessing import Pool

import multiprocessing as mp
from queue import Queue
import threading

import dataset_utils

from spatial_transformer import transformer

### Encoder ###

def encoder(images, feature_maps=16, dilated=False, reuse=False, scope='encoder'):
    with tf.variable_scope(scope, reuse=reuse):
        net = images
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            data_format='NHWC',
                            normalizer_fn=slim.layer_norm,
                            normalizer_params={'scale': False},
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8),
                            biases_initializer=None,
                            activation_fn=tf.nn.relu):

            net = slim.conv2d(net, num_outputs=feature_maps*(2**1), kernel_size=3, scope='encode1/conv3_1')
            end_points['encode1/conv3_1'] = net

            net = slim.avg_pool2d(net, [2, 2], scope='encode1/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**2), kernel_size=3, scope='encode2/conv3_1')
            end_points['encode2/conv3_1'] = net

            net = slim.avg_pool2d(net, [2, 2], scope='encode2/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**3), kernel_size=3, scope='encode3/conv3_1')
            end_points['encode3/conv3_1'] = net

            net = slim.avg_pool2d(net, [2, 2], scope='encode3/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=3, scope='encode4/conv3_1')
            end_points['encode4/conv3_1'] = net

            net = slim.avg_pool2d(net, [2, 2], scope='encode4/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=3, scope='encode5/conv3_1')
            end_points['encode5/conv3_1'] = net

            net = slim.avg_pool2d(net, [2, 2], scope='encode5/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**4), kernel_size=3, scope='encode6/conv3_1')
            end_points['encode6/conv3_1'] = net

            if dilated == False:
                net = slim.avg_pool2d(net, [2, 2], scope='encode6/pool')
            if dilated == False:
                net = slim.conv2d(net, num_outputs=feature_maps*(2**5), kernel_size=2, scope='encode7/conv3_1')
            elif dilated == True:
                net = slim.conv2d(net, num_outputs=feature_maps*(2**5), kernel_size=2, rate=2, scope='encode7/conv3_1')
            end_points['encode7/conv3_1'] = net

            if dilated == False:
                net = slim.avg_pool2d(net, [2, 2], scope='encode7/pool')
            net = slim.conv2d(net, num_outputs=feature_maps*(2**5), kernel_size=1, scope='encode8/conv3_1')
            end_points['encode8/conv3_1'] = net

    return net, end_points


### Decoder ###

#Decoder with skip connections
def decoder(images, encoder_end_points, feature_maps=16, num_classes=2, reuse=False, scope='decoder'):
    def conv(fmaps, ks=3): return lambda net, name: slim.conv2d(net, num_outputs=fmaps, kernel_size=ks, scope=name)
    def conv_t(fmaps, ks=3): return lambda net, name: slim.conv2d_transpose(net, num_outputs=fmaps, kernel_size=ks, stride=[2, 2], scope=name)
    def skip(end_point): return lambda net, name: tf.concat([net, end_point], axis=3, name=name)
    unpool =  lambda net, name: tf.image.resize_nearest_neighbor(net, [2*tf.shape(net)[1], 2*tf.shape(net)[2]], name=name)

    layers = OrderedDict()
    layers['decode8/skip'] = skip(encoder_end_points['encode8/conv3_1'])
    layers['decode8/conv3_1'] = conv(feature_maps*(2**4), ks=1)
    #layers['decode6/unpool'] = unpool

    layers['decode7/skip'] = skip(encoder_end_points['encode7/conv3_1'])
    layers['decode7/conv3_1'] = conv(feature_maps*(2**4), ks=2)
    #layers['decode5/unpool'] = unpool

    layers['decode6/skip'] = skip(encoder_end_points['encode6/conv3_1'])
    layers['decode6/conv3_1'] = conv(feature_maps*(2**4))
    layers['decode6/unpool'] = unpool

    layers['decode5/skip'] = skip(encoder_end_points['encode5/conv3_1'])
    layers['decode5/conv3_1'] = conv(feature_maps*(2**3))
    layers['decode5/unpool'] = unpool

    layers['decode4/skip'] = skip(encoder_end_points['encode4/conv3_1'])
    layers['decode4/conv3_1'] = conv(feature_maps*(2**3))
    layers['decode4/unpool'] = unpool

    layers['decode3/skip'] = skip(encoder_end_points['encode3/conv3_1'])
    layers['decode3/conv3_1'] = conv(feature_maps*(2**2))
    layers['decode3/unpool'] = unpool

    layers['decode2/skip'] = skip(encoder_end_points['encode2/conv3_1'])
    layers['decode2/conv3_1'] = conv(feature_maps*(2**1))
    layers['decode2/unpool'] = unpool

    layers['decode1/skip'] = skip(encoder_end_points['encode1/conv3_1'])
    layers['decode1/classifier'] = lambda net, name: slim.conv2d_transpose(net, num_outputs=num_classes, kernel_size=3, activation_fn=None, scope=name)

    with tf.variable_scope(scope, reuse=reuse):
        net = images
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.layer_norm,
                            normalizer_params={'scale': False},
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8),
                            activation_fn=tf.nn.relu):
            for layer_name, layer_op in layers.items():
                net = layer_op(net, layer_name)
                end_points[layer_name] = net

    return net


### Custom functions ###

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

#Matching filter v8 and v9
def matching_filter(sample, target, mode='depthwise'):
    if mode == 'depthwise':
        conv2ims = lambda inputs : tf.nn.depthwise_conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                      tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                      strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,C
    else:
        conv2ims = lambda inputs : tf.nn.conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,1

    crosscorrelation = tf.map_fn(conv2ims, elems=[sample, target], dtype=tf.float32, name='crosscorrelation_1')

    return crosscorrelation[:, 0, :, :, :] # B,1,H,W,C -> B,H,W,C

### Siamese-U-Net ###

def siamese_u_net(targets, images, feature_maps=24, threshold=0.3):

    #encode target
    targets_encoded, _ = encoder(targets,
                              feature_maps=feature_maps,
                              dilated=False,
                              reuse=False,
                              scope='clean_encoder')


    images_encoded, images_encoded_end_points = encoder(images,
                                                    feature_maps=feature_maps,
                                                    dilated=True,
                                                    reuse=False,
                                                    scope='clutter_encoder')


    #calculate crosscorrelation
    #target_encoded has to be [batch, 1, 1, fmaps] for this to work
    matched = matching_filter(images_encoded, targets_encoded, mode='standard')
    matched = matched * targets_encoded
    # print(matched.get_shape())
    decoder_input = slim.layer_norm(matched, scale=False, center=False, scope='matching_normalization')
    # print(decoder_input.get_shape())

    #get segmentation mask
    segmentations = decoder(decoder_input,
                              images_encoded_end_points,
                              feature_maps=feature_maps,
                              reuse=False,
                              scope='decoder')

    return segmentations



### MaskNet ###

def mask_net(targets, images, labels=None, feature_maps=24, training=False, threshold=0.3, rgb_mean=127, rgb_std=127):

    #encode target
    targets_encoded, _ = encoder(targets,
                              feature_maps=feature_maps,
                              dilated=False, reuse=False,
                              scope='clean_encoder')

    images_encoded, images_encoded_end_points = encoder(images,
                                                    feature_maps=feature_maps,
                                                    dilated=True, reuse=False,
                                                    scope='clutter_encoder')

    #calculate crosscorrelation
    #target_encoded has to be [batch, 1, 1, fmaps] for this to work
    matched = matching_filter(images_encoded, targets_encoded, mode='standard')
    matched = matched * targets_encoded

    # Get size of matching
    isz = images.get_shape().as_list()
    ix = isz[1]
    iy = isz[2]
    tsz = targets.get_shape().as_list()
    tx = tsz[1]
    ty = tsz[2]
    msz = matched.get_shape().as_list()
    batch_size = msz[0]
    mx = msz[1]
    my = msz[2]

    # Generate proposals
    # There are 3 modes:
    # Training the encoder and decoder
    # Training the discriminator
    # Evaluation
    if training != False:

        # Get center of mass of labels to determine fg proposals
        comx, comy = center_of_mass(labels)
        comx = comx/(ix/mx)
        comy = comy/(iy/my)

        # Initialize fg & bgindices
        fg_index = [0 for x in range(4)]
        # select the 4 locations around the label com as fg proposals
        fg_index[0] = tf.cast(tf.floor(comx) + my * tf.floor(comy), tf.int32)
        fg_index[1] = tf.cast(tf.ceil(comx) + my * tf.floor(comy), tf.int32)
        fg_index[2] = tf.cast(tf.floor(comx) + my * tf.ceil(comy), tf.int32)
        fg_index[3] = tf.cast(tf.ceil(comx) + my * tf.ceil(comy), tf.int32)
        # Draw random indices for bg proposals
        bg_index = [tf.expand_dims(tf.random_shuffle(tf.range(0, mx*my)), axis=1) for x in range(batch_size)]
        bg_index = tf.concat(bg_index, axis=1)

        if training == 'encoder_decoder':
            # Generate 4 foreground and 4 background proposals
            num_proposals = 8
            proposal_range = range(num_proposals)

            # create index
            index = [0 for x in proposal_range]
            for i in range(4):
                index[i] = fg_index[i]
            # Select 4 random locations for bg proposals
            for i in range(4,num_proposals):
                index[i] = bg_index[i,:]
                if any([index[i] == index[j] for j in range(4)]):
                    index[i] = bg_index[i+(num_proposals - 4),:]

        elif training == 'discriminator':
            # Generate x foreground and y background proposals
            num_proposals = 4
            proposal_range = range(num_proposals)

            fg_index = tf.stack(fg_index, axis=0)
            random_fg_index = tf.random_shuffle(tf.range(0, 4))

            index = [0 for x in proposal_range]
            index[0] = fg_index[random_fg_index[0],:]
            for l in range(num_proposals-1):
                index[l+1] = bg_index[l,:]


            tensor_index = tf.expand_dims(tf.stack(index, axis=0), axis=-1)
            tensor_labels = tf.expand_dims(tf.stack([1, 0, 0, 0], axis=0), axis=1)
            tensor_labels = tf.expand_dims(tf.tile(tensor_labels, [1, batch_size]), axis=-1)

            shuffled_index_and_labels = tf.random_shuffle(tf.concat([tensor_index, tensor_labels], axis=2))

            index = shuffled_index_and_labels[...,0]
            labels = tf.cast(tf.transpose(shuffled_index_and_labels[...,1]), tf.float32)

    else:
        proposal_range = range(mx*my)
        index = [tf.ones(batch_size, dtype=tf.int32) * q for q in proposal_range]

    # Run Decoder with proposals
    proposed_segmentations = [0 for x in proposal_range]
    scores = [0 for x in proposal_range]
    for q in proposal_range:

        # To share weights between proposals they have to be inititalized
        # for the first proposal and reused afterwards
        if q == 0:
            reuse = False
        else:
            reuse = True

        # Generate the one-hot proposal
        one_hot_proposal = tf.one_hot(index[q], mx*my)
        mask = tf.reshape(one_hot_proposal, [batch_size,mx,my,1])
        # Apply the proposal
        masked = matched * mask


        decoder_input = slim.layer_norm(masked, scale=False, center=False, scope='matching_normalization')

        #get segmentation mask
        segmentation = decoder(decoder_input,
                                  images_encoded_end_points,
                                  feature_maps=feature_maps,
                                  reuse=reuse,
                                  scope='decoder')
        proposed_segmentations[q] = segmentation

        if training == 'encoder_decoder':
            continue
        #Threshold segmentations
        binary_segmentation = threshold_segmentations(segmentation, threshold=threshold)

        # Calculate offset for crop
        shift = get_shift_xy(binary_segmentation)
        theta_0 = tf.tile([[1/3, 0, 0, 0, 1/3, 0]], [batch_size, 1])
        theta_shift = tf.transpose(tf.scatter_nd([[2], [5]], shift, [6, batch_size]))
        theta = theta_0 + theta_shift
        # Crop segmentation mask
        crop = transformer(binary_segmentation, theta, out_size=[tx,ty])
        crop.set_shape([batch_size,tx,ty,1])
        # Center and Normalize the crop for the discriminator
        crop = crop * 255
        crop = (crop - rgb_mean)/rgb_std
        # Map crop to RGB space
        crop = tf.tile(crop, [1,1,1,3])

        crop_encoded, _ = encoder(crop,
                           feature_maps=feature_maps,
                           dilated=False, reuse=reuse,
                           scope='discriminator')

        l1 = tf.abs(crop_encoded - targets_encoded)
        score = slim.fully_connected(l1, num_outputs=1,
                                     biases_initializer=None,
                                     reuse=reuse, activation_fn=tf.sigmoid,
                                     scope='weighted_sum')
        score = tf.squeeze(score, axis=[1,2,3])
        scores[q] = score

    # Select highest ranking segmentation mask
    segmentations = tf.stack(proposed_segmentations, axis=1)
    if not training == 'encoder_decoder':
        scores = tf.stack(scores, axis=1)
        indices = tf.stack((tf.range(batch_size), tf.argmax(scores, axis=1, output_type=tf.int32)), axis=1)
        segmentations = tf.gather_nd(segmentations, indices)

    if training == 'discriminator':
        return segmentations, scores, labels,
    else:
        return segmentations

### Utils ###

# def load_dataset_train(dataset_dir, memory_mapping=False):
#     ims_train = []
#     seg_train = []
#     tar_train = []
#     for fold_num in range(NUM_FOLDS):
#         fold_name = "fold_{:04d}".format(fold_num)
#         ims, seg, tar = load_fold_train(
#             os.path.join(dataset_dir, fold_name), memory_mapping)
#         ims_train.append(ims)
#         seg_train.append(seg)
#         tar_train.append(tar)

#     return np.concatenate(ims_train), np.concatenate(seg_train), np.concatenate(tar_train)


def load_train(fold_dir, index, memory_mapping=False):
    path = os.path.join(fold_dir, 'train/')
    if memory_mapping == True:
        ims_train = np.load(
            os.path.join(path, 'image_{:08d}.npy'.format(index)), mmap_mode='r')
        seg_train = np.load(
            os.path.join(path, 'segmentation_{:08d}.npy'.format(index)), mmap_mode='r')
        tar_train = np.load(os.path.join(path, 'target_{:08d}.npy'.format(index)), mmap_mode='r')
    else:
        ims_train = np.load(
            os.path.join(path, 'image_{:08d}.npy'.format(index)))
        seg_train = np.load(
            os.path.join(path, 'segmentation_{:08d}.npy'.format(index)))
        tar_train = np.load(os.path.join(path, 'target_{:08d}.npy'.format(index)))

    return ims_train, seg_train, tar_train


# def load_dataset_val(dataset_dir, subset):
#     ims_val = []
#     seg_val = []
#     tar_val = []
#     for fold_num in range(NUM_FOLDS):
#         fold_name = "fold_{:04d}".format(fold_num)
#         ims, seg, tar = load_fold_val(
#             os.path.join(dataset_dir, fold_name), subset)
#         ims_val.append(ims)
#         seg_val.append(seg)
#         tar_val.append(tar)
#     return np.concatenate(ims_val), np.concatenate(seg_val), np.concatenate(tar_val)


def load_val(fold_dir, index, subset):
    if subset == 'train':
        #load val_train data
        path = os.path.join(fold_dir, 'val-train/')
    elif subset == 'eval':
        #load val_eval data
        path = os.path.join(fold_dir, 'val-one-shot')
    else:
        print(subset + ' is not a valid subset')
    ims_val = np.load(
        os.path.join(path, 'image_{:08d}.npy'.format(index)))
    seg_val = np.load(
        os.path.join(path, 'segmentation_{:08d}.npy'.format(index)))
    tar_val = np.load(
        os.path.join(path, 'target_{:08d}.npy'.format(index)))

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


def load_test(fold_dir, index, subset):
    if subset == 'train':
        #load test_train data
        path = os.path.join(fold_dir, 'test-train/')
    elif subset == 'eval':
        #load test_etest data
        path = os.path.join(fold_dir, 'test-one-shot/')
    else:
        print(subset + ' is not a testid subset')

    ims_test = np.load(
        os.path.join(path, 'image_{:08d}.npy'.format(index)))
    seg_test = np.load(
        os.path.join(path, 'segmentation_{:08d}.npy'.format(index)))
    tar_test = np.load(
        os.path.join(path, 'target_{:08d}.npy'.format(index)))

    return ims_test, seg_test, tar_test


def threaded_batch_generator(generator, batch_size, max_queue_len=1):
    queue = Queue(maxsize=max_queue_len)
    sentinel = object()

    def producer():
        # expect three numpy arrays "blocks"
        ims_batch, seg_batch, tar_batch = [], [], []
        index = 0
        for ims, seg, tar in generator:
            perms = np.random.permutation(ims.shape[0])
            for i in range(ims.shape[0]):
                ims_batch.append(ims[perms[i]])
                seg_batch.append(seg[perms[i]])
                tar_batch.append(tar[perms[i]])
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
        num_blocks,
        perms,
        step=0):
    """Yields mini-batches for training/testing purposes.

    Args:
      dataset_dir: The direct folder where images/segs/targets are held.
      split: "test", "train", or "val"
      modifier: control flow parameter for load_X function above
    """
    while True:
        index = perms[step]
        if split == "train":
            im, seg, tar = load_train(dataset_dir, index, modifier)
        if split == "val":
            im, seg, tar = load_val(dataset_dir, index, modifier)
        if split == "test":
            im, seg, tar = load_test(dataset_dir, index, modifier)

        # make seg binary
        seg[seg > 0] = 1
        yield im, seg, tar
        # images_batch = np.zeros((batch_size, IMAGE_DIM, IMAGE_DIM, 3))
        # labels_batch = np.zeros((batch_size, IMAGE_DIM, IMAGE_DIM, 1))
        # target_batch = np.zeros((batch_size, TARGET_DIM, TARGET_DIM, 3))
        # if not multi:
        #     for i in range(batch_size):
        #         index = perms[step*batch_size+i]
        #         # load image based on generator type
        #         if split == "train":
        #             im, seg, tar = load_train(dataset_dir, index, modifier)
        #         if split == "val":
        #             im, seg, tar = load_val(dataset_dir, index, modifier)
        #         if split == "test":
        #             im, seg, tar = load_test(dataset_dir, index, modifier)
        #         images_batch[i, :, :, :] = im
        #         labels_batch[i, :, :, :] = seg
        #         target_batch[i, :, :, :] = tar
        #     yield images_batch, labels_batch, target_batch
        #     step += 1
        # if multi:
        #     results = Parallel(n_jobs=-1, verbose=0)(delayed(batch_helper)(dataset_dir, split, modifier, batch_size, perms, s) for s in range(step, step + 10))
        #     example_count = 0
        #     for i in range(len(results)):
        #         yield results[i][0], results[i][1], results[i][2]
        #         step += 1
        step += 1
        if step == num_blocks:
            step = 0

# def make_batch(batch_size, ims, seg, tar, perms=np.zeros(10), step=0):
#     images_batch = np.zeros((batch_size,ims.shape[1],ims.shape[2],3))
#     labels_batch = np.zeros((batch_size,ims.shape[1],ims.shape[2],1))
#     target_batch = np.zeros((batch_size,tar.shape[1],tar.shape[2],3))

#     if all(perms == 0):
#         perms = np.random.permutation(tar.shape[0])

#     ntarget_index = np.random.randint(tar.shape[1])

#     for i in range(batch_size):
#         index = perms[step*batch_size+i]
#         images_batch[i,:,:,:] = ims[index,:,:,:]
#         labels_batch[i,:,:,:] = seg[index,:,:,:]
#         target_batch[i,:,:,:] = tar[index,:,:,:]

#     return images_batch, target_batch, labels_batch

def threshold_segmentations(segmentations, threshold=0.3):
    seg_softmax = tf.nn.softmax(segmentations, axis=-1)
    seg = tf.cast(seg_softmax[...,1] > threshold, tf.int32)
    seg = tf.expand_dims(seg, axis=-1)

    return seg

#IoU claculation routine
def calculate_IoU(segmentations, labels, threshold=0.3):
    pred = tf.squeeze(labels, axis=-1)
    seg = tf.squeeze(segmentations, axis=-1)
    IoU = tf.reduce_sum(pred*seg, axis=(1,2))/(tf.reduce_sum(pred, axis=(1,2))+tf.reduce_sum(seg, axis=(1,2))-tf.reduce_sum(pred*seg, axis=(1,2)))
    clean_IoU = tf.where(tf.is_nan(IoU), tf.ones_like(IoU), IoU) #Remove NaNs which appear when the target does not exist

    return clean_IoU


### Training ###

def training(dataset_dir,
             logdir,
             epochs,
             train_size,
             val_size,
             block_size,
             model='siamese-u-net',
             train_mode='encoder_decoder',
             feature_maps=24,
             batch_size=250,
             learning_rate=0.0005,
             pretraining_checkpoint=None,
             maximum_number_of_steps=0):

    # Currently only the siamese-u-net is implemented
    assert model in ['siamese-u-net', 'mask-net']

    print("Drawing from {}".format(dataset_dir))

    #Shuffle samples
    perms = np.random.permutation(train_size // block_size)
    val_perms = np.random.permutation(val_size // block_size)

    with tf.Graph().as_default():

        #Define logging parameters
        t = time.time()
        OSEG_CKPT_FILE = logdir + 'Run.ckpt'
        if pretraining_checkpoint is not None:
            PRETRAINING_CKPT_FILE = pretraining_checkpoint + 'Run.ckpt'
        weight_logging = True
        if not tf.gfile.Exists(logdir):
            tf.gfile.MakeDirs(logdir)

        #Define training parameters
        batch_size = batch_size
        print("Batch size: {}".format(batch_size))

        # max_steps for epoch training
        # num_blocks for loading block data
        max_train_steps = train_size // batch_size
        num_train_blocks = train_size // block_size
        max_val_steps = val_size // batch_size
        num_val_blocks = val_size // block_size

        if maximum_number_of_steps != 0:
            print('Going to run for %.d steps'%(np.min([epochs * max_train_steps, maximum_number_of_steps])))
        else:
            print('Going to run for %.d steps'%(epochs * max_train_steps))

        # Get dataset information and statistics
        # Generate batch for this purpose
        # Sample mean from a large number of points
        mean, std = 0.0, 0.0
        sample_im_batch = []
        sample_generator = block_generator(dataset_dir,
                                           split="train",
                                           modifier=False,
                                           num_blocks=10,
                                           perms=perms)
        for i in range((1000 // block_size) + 1):
            sample = next(sample_generator)
            im_block, _, tar_block = sample
            sample_im_batch.append(im_block)

        mean = np.mean(np.array(sample_im_batch))
        std = np.std(np.array(sample_im_batch))

        #generate tensorflow placeholders and variables
        images = tf.placeholder(tf.float32, shape=[batch_size,im_block.shape[1],im_block.shape[2],3], name='images')
        labels = tf.placeholder(tf.int32, shape=[batch_size,im_block.shape[1],im_block.shape[2],1], name='labels')
        targets = tf.placeholder(tf.float32, shape=[batch_size,tar_block.shape[1],tar_block.shape[2],3], name='targets')
        learn_rate = tf.Variable(learning_rate)

        #preprocess images - removed due to rotated images not meaning well.
        # targets = (targets - mean)/std
        # images = (images - mean)/std

        #get predictions
        if model == 'siamese-u-net':
            segmentations = siamese_u_net(targets, images, feature_maps=feature_maps)
            final_labels = labels
        elif model == 'mask-net':
            if train_mode == 'encoder_decoder':
                segmentations = mask_net(targets, images, labels, feature_maps=feature_maps, training=train_mode)
                final_labels = tf.tile(tf.expand_dims(labels, axis=1), [1, 4, 1, 1, 1])
                final_labels = tf.concat([final_labels, tf.zeros_like(final_labels)], axis=1)
            elif train_mode == 'discriminator':
                segmentations, scores, score_labels = mask_net(targets, images, labels,
                                                               feature_maps=feature_maps, training=train_mode,
                                                               rgb_mean=mean, rgb_std=std)
                final_labels = labels

        #Update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #Define Losses
        if model == 'mask-net' and train_mode == 'discriminator':
            epsilon = 0.0001
            loss_true = -tf.reduce_mean(score_labels*tf.log(scores+epsilon))
            loss_false = -tf.reduce_mean((1-score_labels)*tf.log(1-scores+epsilon))
            main_loss = loss_true + loss_false
            reg_loss = tf.add_n(tf.losses.get_regularization_losses(scope='discriminator'))
        else:
            main_loss = tf.losses.sparse_softmax_cross_entropy(labels=final_labels, logits=segmentations, scope='losses')
            reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        loss = main_loss + reg_loss

        #Get encoder and decoder variables
        train_var_list = None # Train all variables by default
        if model == 'mask-net':
            clean_encoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clean_encoder')
            clutter_encoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clutter_encoder')
            decoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
            if train_mode == 'discriminator':
                discriminator_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
                weighted_sum_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='weighted_sum')
                # For discriminator training train only the discriminator
                train_var_list = [discriminator_var_list, weighted_sum_var_list]

        #Training step
        optimizer = tf.train.AdamOptimizer(learn_rate)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=train_var_list)


        #Calculate metrics: IoU
        if model == 'mask-net' and train_mode == 'encoder_decoder':
            ssz = segmentations.get_shape().as_list()
            lsz = final_labels.get_shape().as_list()
            segmentations = tf.reshape(segmentations, [-1, ssz[2], ssz[3], ssz[4]])
            final_labels = tf.reshape(final_labels, [-1, lsz[2], lsz[3], lsz[4]])
        binary_segmentations = threshold_segmentations(segmentations)
        mean_IoU = tf.reduce_mean(calculate_IoU(binary_segmentations, final_labels))

        #Create summaries
        tf.summary.scalar('main_loss', main_loss)
        tf.summary.scalar('regularization_loss', reg_loss)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('mean_IoU', mean_IoU)

        #Collect summaries
        summary = tf.summary.merge_all()

        #Logging
        def name_in_checkpoint(var):
            if "discriminator" in var.op.name:
                return var.op.name.replace("discriminator", "clean_encoder")

        saver = tf.train.Saver(max_to_keep=None)
        restorer = tf.train.Saver(slim.get_model_variables())
        # Initialize the discriminator with the clean encoder parameters
        if model == 'mask-net' and train_mode == 'discriminator':
            restorer = tf.train.Saver(slim.get_variables_to_restore(exclude=['discriminator', 'weighted_sum']))
            variables_for_init = slim.get_variables_to_restore(include=['discriminator'])
            variables_for_init = {name_in_checkpoint(var):var for var in variables_for_init}
            discriminator_initializer = tf.train.Saver(variables_for_init)

        # Initalize batch generators
        train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="train",
                modifier=False,
                num_blocks=num_train_blocks,
                perms=perms), batch_size, max_queue_len=100)
        val_train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="val",
                modifier="train",
                num_blocks=num_val_blocks,
                perms=val_perms), batch_size)
        val_eval_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="val",
                modifier="eval",
                num_blocks=num_val_blocks,
                perms=val_perms), batch_size)

        print("Beginning session")

        #Start Session
        with tf.Session() as sess:
            #Initialize logging files
            summary_writer_train = tf.summary.FileWriter(logdir, sess.graph)
            summary_writer_val_train = tf.summary.FileWriter(logdir + 'val_train')
            summary_writer_val_eval = tf.summary.FileWriter(logdir + 'val_eval')

            #Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            step_count = 0
            if pretraining_checkpoint is not None:
                restorer.restore(sess, PRETRAINING_CKPT_FILE)
                print("Restoring from {}".format(PRETRAINING_CKPT_FILE))
                if model == 'mask-net' and train_mode == 'discriminator':
                    discriminator_initializer.restore(sess, PRETRAINING_CKPT_FILE)

            #Training loop
            print('Starting to train')
            duration = []
            losses = []
            for epoch in range(epochs):
                print("Epoch {}/{}".format(epoch + 1, epochs))
                print("Training for {} steps".format(max_train_steps))
                #Learning rate schelude
                # if epoch == epochs//2 or epoch == 3*epochs//4 or epoch == 7*epochs//8:
                if len(losses) > 0 and max(losses) - min(losses) < 0.001:
                    learning_rate = learning_rate/2
                    print('lowering learning rate to %.4f'%(learning_rate))
                losses = []
                #Run trainings step
                # for step in range(max_steps):
                last_time = time.time()
                for step in range(max_train_steps):
                    images_batch, labels_batch, target_batch = next(train_generator)
                    _, loss_value = sess.run([train_op, loss],
                                             feed_dict = {targets: target_batch,
                                                          images: images_batch,
                                                          labels: labels_batch,
                                                          learn_rate: learning_rate})
                    step_count += 1
                    step += 1
                    new_time = time.time()
                    duration.append(new_time - last_time)
                    last_time = new_time


                    #Evaluate
                    if step_count % 200 == 0 or step_count == 1:
                        #Evaluate and print training error and IoU
                        summary_str_train, tf_IoU = sess.run([summary, mean_IoU],
                                                             feed_dict = {targets: target_batch,
                                                                          images: images_batch,
                                                                          labels: labels_batch})
                        summary_writer_train.add_summary(summary_str_train, step_count)
                        summary_writer_train.flush()
                        print('Step %5d: loss = %.4f mIoU: %.3f (%.3f sec)'
                              % (step_count, np.mean(loss_value), tf_IoU, np.mean(duration)))
                        duration = []
                        losses.append(np.mean(loss_value))

                        #evaluate val_train
                        images_batch, labels_batch, target_batch = next(val_train_generator)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch,
                                                                   images: images_batch,
                                                                   labels: labels_batch})
                        summary_writer_val_train.add_summary(summary_str, step_count)
                        summary_writer_val_train.flush()

                        #evaluate val_eval
                        images_batch, labels_batch, target_batch = next(val_eval_generator)
                        summary_str = sess.run(summary, feed_dict={targets: target_batch,
                                                                   images: images_batch,
                                                                   labels: labels_batch})

                        summary_writer_val_eval.add_summary(summary_str, step_count)
                        summary_writer_val_eval.flush()


                    #Create checkpoint
                    if step_count % 400 == 0 or step_count == epochs*max_train_steps:
                        checkpoint_file = os.path.join(logdir, 'Run_Epoch{}_Step{}.ckpt'.format(epoch, step_count))
                        saver.save(sess, checkpoint_file)
                        checkpoint_file = os.path.join(logdir, 'Run.ckpt')
                        saver.save(sess, checkpoint_file)


        print('Total time: ', time.time()-t)



### Evaluate ###

#Network training
def evaluation(dataset_dir,
               logdir,
               test_size,
               block_size,
               model='siamese-u-net',
               feature_maps=24,
               batch_size=250,
               threshold=0.3,
               max_steps=0,
               vis=False):

    # Currently only the siamese-u-net is implemented
    assert model in ['siamese-u-net', 'mask-net']

    with tf.Graph().as_default():

        #Define logging parameters
        OSEG_CKPT_FILE = logdir + 'Run_Epoch4_Step117200.ckpt'

        perms = np.random.permutation(test_size // block_size)

        # #Load dataset
        # print('Loading dataset: ' + dataset_dir)
        # ims_val_train, seg_val_train, tar_val_train = load_dataset_test(dataset_dir, subset='train')
        # ims_val_eval, seg_val_eval, tar_val_eval = load_dataset_test(dataset_dir, subset='eval')
        # print('Done loading dataset')

        #Define training parameters
        batch_size = batch_size
        max_steps = test_size // batch_size
        num_blocks = test_size // block_size

        # Get dataset information and statistics
        # Generate batch for this purpose
        # Sample mean from at most 1000 data points
        mean, std = 0.0, 0.0
        sample_im_batch = []
        sample_generator = block_generator(dataset_dir,
                                           split="test",
                                           modifier="train",
                                           num_blocks=10,
                                           perms=perms)
        for i in range((10 // block_size) + 1):
            sample = next(sample_generator)
            im_block, _, tar_block = sample
            sample_im_batch.append(im_block)

        mean = np.mean(np.array(sample_im_batch))
        std = np.std(np.array(sample_im_batch))
        im_block = np.ones((1, 384, 384, 1))
        tar_block = np.ones((1, 128, 128, 1))

        #generate tensorflow placeholders and variables
        images = tf.placeholder(tf.float32, shape=[batch_size,im_block.shape[1],im_block.shape[2],3], name='images')
        labels = tf.placeholder(tf.int32, shape=[batch_size,im_block.shape[1],im_block.shape[2],1], name='labels')
        targets = tf.placeholder(tf.float32, shape=[batch_size,tar_block.shape[1],tar_block.shape[2],3], name='targets')

        #preprocess images
        # targets = (targets - mean)/std
        # images = (images - mean)/std

        #get predictions
        if model == 'siamese-u-net':
            segmentations = siamese_u_net(targets, images, feature_maps=feature_maps)
        elif model == 'mask-net':
            segmentations = mask_net(targets, images, feature_maps=feature_maps, training=False, rgb_mean=mean, rgb_std=std)

        #Calculate metrics: IoU
        binary_segmentations = threshold_segmentations(segmentations)

        mean_IoU = calculate_IoU(binary_segmentations, labels, threshold=threshold)
        #Calculate metrics: Localization Accuracy
        lcomx, lcomy = center_of_mass(labels)
        comx, comy = center_of_mass(binary_segmentations)
        euclidian_distance = tf.sqrt((lcomx-comx)**2 + (lcomy-comy)**2)
        distance_metric = tf.cast(euclidian_distance < 5, tf.float32)
        mean_dist = tf.reduce_mean(distance_metric)

        #Logging
        saver = tf.train.Saver()
        restorer = tf.train.Saver(slim.get_model_variables())

        test_train_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="test",
                modifier="train",
                num_blocks=num_blocks,
                perms=perms), batch_size)
        test_eval_generator = threaded_batch_generator(
            block_generator(
                dataset_dir,
                split="test",
                modifier="eval",
                num_blocks=num_blocks,
                perms=perms), batch_size)

        #Start Session
        with tf.Session() as sess:
            seg_counter = 0
            #Initialize from scratch or finetune from previous training
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, OSEG_CKPT_FILE)

            #Run trainings step
            val_IoU = [0 for x in range(max_steps)]
            os_IoU = [0 for x in range(max_steps)]
            val_distances = [0 for x in range(max_steps)]
            os_distances = [0 for x in range(max_steps)]
            percents = []
            IoUs = []
            for step in tqdm(range(max_steps)):
                images_batch, labels_batch, target_batch = next(test_train_generator)
                val_IoU[step], val_distances[step], segs = sess.run(
                    [mean_IoU, mean_dist, binary_segmentations],
                    feed_dict = {targets: target_batch,
                                 images: images_batch,
                                 labels: labels_batch})
                for seg_idx in range(labels_batch.shape[0]):
                    seg = labels_batch[seg_idx]
                    percent = len(np.where(seg > 0)[0]) / (seg.shape[0] * seg.shape[1])
                    percents.append(percent)
                    IoUs.append(val_IoU[step][seg_idx])
                    # print(percent, val_IoU[step][seg_idx])
                    seg_counter += 1

                # Code to plot the segmentation v. IoU graph
                # if seg_counter == 1000:
                #     xs, ys = zip(*sorted(zip(percents, IoUs)))
                #     plt.plot(xs, ys)
                #     plt.savefig("./fig")
                #     break

                if vis and step < 100:
                    saved_idx = np.random.choice(batch_size)
                    im = images_batch[saved_idx]
                    seg = labels_batch[saved_idx]
                    pred = segs[saved_idx]
                    vis_dir = os.path.join(logdir, "vis/")
                    dataset_utils.mkdir_if_missing(vis_dir)
                    plt.figure(figsize=(15, 15))
                    plt.imshow(im)
                    plt.imshow(np.ma.masked_where(seg[...,0] == 0, seg[...,0]), cmap="summer")
                    plt.title("Ground Truth Segmask", {"fontsize": 16})
                    plt.savefig(os.path.join(vis_dir, "gt_overlay_{}.png").format(step))
                    plt.clf()

                    plt.imshow(im)
                    plt.imshow(np.ma.masked_where(pred[...,0] == 0, pred[...,0]), cmap="coolwarm")
                    plt.title("Predicted Segmask", {"fontsize": 16})
                    plt.savefig(os.path.join(vis_dir, "pred_overlay_{}.png").format(step))
                    plt.clf()

                    np.save(
                        os.path.join(vis_dir, "sample_pred_{}.npy").format(step),
                        segs[saved_idx],
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_gt_{}.npy").format(step),
                        labels_batch[saved_idx],
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_tar_{}.npy").format(step),
                        target_batch[saved_idx],
                        allow_pickle=False)
                    np.save(
                        os.path.join(vis_dir, "sample_im_{}.npy").format(step),
                        images_batch[saved_idx],
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
