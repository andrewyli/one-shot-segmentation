import tensorflow as tf
import utils

slim = tf.contrib.slim

from collections import OrderedDict, deque


class Model():
    """(abstract) Base class for models.
    """

    def generate_segmentations(self, targets, images, feature_maps=24, threshold=0.3):
        """Returns the segmentations (N x H x W x 2) in the form of sigmoid outputs.
        """
        pass

class SiameseUNet(Model):
    """A Model class for Siamese U-Net.
    """

    def __init__(self, mode="train", reg_factor=0.0, dropout=0.0):
        self.mode = mode
        self.reg_factor = reg_factor
        self.dropout = dropout
        print("Initialized model with mode {}.".format(mode))

    ### Encoder ###
    def encoder(self, images, feature_maps=16, dilated=False, reuse=False, scope='encoder'):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            net = images
            end_points = OrderedDict()
            combined_reg = tf.contrib.layers.sum_regularizer([
                tf.contrib.layers.l2_regularizer(2e-6),
                utils.rotation_regularizer(self.reg_factor)
            ])
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                data_format='NHWC',
                                normalizer_fn=slim.layer_norm,
                                normalizer_params={'scale': False},
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'),
                                weights_regularizer=combined_reg,
                                biases_initializer=None,
                                activation_fn=tf.nn.relu):

                net = slim.conv2d(net, num_outputs=feature_maps*(2**2), kernel_size=3, scope='encode1/conv3_1')
                end_points['encode1/conv3_1'] = net

                net = slim.avg_pool2d(net, [2, 2], scope='encode1/pool')
                net = slim.conv2d(net, num_outputs=feature_maps*(2**3), kernel_size=3, scope='encode2/conv3_1')
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

                # apply dropout
                if self.mode == "train":
                    net = tf.nn.dropout(net, rate=self.dropout)
                else:
                    net = tf.nn.dropout(net, rate=0.0)

                end_points['encode8/conv3_1'] = net

        return net, end_points


    ### Decoder ###

    #Decoder with skip connections
    def decoder(self, images, encoder_end_points, feature_maps=16, num_classes=2, reuse=False, scope='decoder'):
        def conv(fmaps, ks=3): return lambda net, name: slim.conv2d(net, num_outputs=fmaps, kernel_size=ks, scope=name)
        def conv_t(fmaps, ks=3): return lambda net, name: slim.conv2d_transpose(net, num_outputs=fmaps, kernel_size=ks, stride=[2, 2], scope=name)
        def skip(end_point): return lambda net, name: tf.concat([net, end_point], axis=3, name=name)
        unpool =  lambda net, name: tf.compat.v1.image.resize_nearest_neighbor(net, [2*tf.shape(net)[1], 2*tf.shape(net)[2]], name=name)

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
        # layers['decode1/convt'] = lambda net, name: slim.conv2d_transpose(net, num_outputs=feature_maps, kernel_size=3, activation_fn=None, scope=name)

        # layers['decode0/classifier'] = lambda net, name: slim.conv2d_transpose(net, num_outputs=num_classes, kernel_size=3, activation_fn=None, scope=name)

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            net = images
            end_points = OrderedDict()
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                normalizer_fn=slim.layer_norm,
                                normalizer_params={'scale': False},
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG'),
                                # weights_regularizer=tf.contrib.layers.l2_regularizer(5e-7),
                                activation_fn=tf.nn.relu):
                for layer_name, layer_op in layers.items():
                    net = layer_op(net, layer_name)
                    end_points[layer_name] = net

        return net

    #Matching filter v8 and v9
    def matching_filter(self, sample, target, mode='depthwise'):
        if mode == 'depthwise':
            conv2ims = lambda inputs : tf.nn.depthwise_conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                          tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                          strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,C
        else:
            conv2ims = lambda inputs : tf.nn.conv2d(tf.expand_dims(inputs[0], 0),  # H,W,C -> 1,H,W,C
                                                    tf.expand_dims(inputs[1], 3),  # H,W,C -> H,W,C,1
                                                    strides=[1,1,1,1], padding="SAME") # Result of conv is 1,H,W,1

        crosscorrelation = tf.map_fn(conv2ims, elems=[sample, target],dtype=tf.float32, name='crosscorrelation_1')

        return crosscorrelation[:, 0, :, :, :] # B,1,H,W,C -> B,H,W,C

    ### Siamese-U-Net ###

    def generate_segmentations(self, targets, images, feature_maps=24, threshold=0.3):

        #encode target
        targets_encoded, _ = self.encoder(
            targets,
            feature_maps=feature_maps,
            dilated=False,
            reuse=False,
            scope='clean_encoder')

        images_encoded, images_encoded_end_points = self.encoder(
            images,
            feature_maps=feature_maps,
            dilated=True,
            reuse=False,
            scope='clutter_encoder')

        # print(images_encoded.get_shape().as_list(), targets_encoded.get_shape().as_list())
        # calculate crosscorrelation
        # target_encoded has to be [batch, 1, 1, fmaps] for this to work
        matched = self.matching_filter(
            images_encoded, targets_encoded, mode='standard')
        matched = matched * targets_encoded
        # print(matched.get_shape())
        decoder_input = slim.layer_norm(matched, scale=False, center=False, scope='matching_normalization')
        # print(decoder_input.get_shape())

        # get segmentation mask
        segmentations = self.decoder(
            decoder_input,
            images_encoded_end_points,
            feature_maps=feature_maps,
            reuse=False,
            scope='decoder')

        return segmentations
