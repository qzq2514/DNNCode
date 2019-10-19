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
                net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,16,16,32]

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,8,8,64]

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, padding='SAME')  # [None,4,4,128]
            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
                net = tf.nn.relu6(net)
                net = tf.reduce_mean(net, axis=[1, 2], name="global_pool", keep_dims=True)  # [None,1,1,256]

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 28, [1, 1], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.layers.batch_normalization(net, training=is_training, reuse=reuse)
            net = tf.contrib.layers.flatten(net)
            print("net_out:", net)

        return net

    def loss(self,left_faltten,right_flatten,pair_label,margin):
        dist = tf.reduce_sum(tf.square(left_faltten-right_flatten),1)
        pos_dist = pair_label*dist    #同类样本标签为1
        neg_dist = (1-pair_label)*tf.maximum(margin-dist,0) #同类样本标签为0

        return tf.reduce_mean(pos_dist+neg_dist)/2,dist