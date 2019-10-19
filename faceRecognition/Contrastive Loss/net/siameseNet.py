import tensorflow.contrib.slim as slim
import tensorflow as tf

class siameseNet():
    def __init__(self):
        pass

    def pre_process(self,input):
        preprocessed_input = tf.to_float(input)
        preprocessed_input = tf.subtract(preprocessed_input, 128.0)
        preprocessed_input = tf.div(preprocessed_input, 255.0)

        red,green,blue=tf.split(preprocessed_input,num_or_size_splits=3,axis=3)
        preprocessed_input = (tf.multiply(blue, 0.2989) + tf.multiply(green, 0.5870) +
                              tf.multiply(red, 0.1140))
        return preprocessed_input

    def inference(self,input,reuse=False,is_training=True):
        preprocessed_inputs = self.pre_process(input)

        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(preprocessed_inputs, 32, [5, 5], activation_fn=None,
                                               padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net=tf.layers.batch_normalization(net,training=is_training,reuse=reuse)
                net=tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,16,16,32]

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,8,8,64]

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,4,4,128]
            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.reduce_mean(net, axis=[1, 2], name="global_pool", keep_dims=True)  # [None,1,1,256]

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 14, [1, 1], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                # net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
            net = tf.contrib.layers.flatten(net)
            print("net_out:", net)
        return net

    '''def siamese_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.90,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            "activation_fn":tf.nn.relu
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.avg_pool2d,slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc'''

    def loss(self,left_faltten,right_flatten,pair_label,margin):
        dist=tf.sqrt(tf.reduce_sum(tf.square(left_faltten-right_flatten),1))
        similarity=pair_label*tf.square(dist)
        disimilarity=(1-pair_label)*tf.square(tf.maximum(margin-dist,0))

        return tf.reduce_mean(similarity+disimilarity)/2