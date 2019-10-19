import tensorflow.contrib.slim as slim
import tensorflow as tf
from TripletLossTool import TripletLossTool

class siameseNet():
    def __init__(self):
        self.triplet_loss_tool=TripletLossTool()

    def pre_process(self,input):
        preprocessed_input = tf.to_float(input)
        preprocessed_input = tf.subtract(preprocessed_input, 127.5)
        preprocessed_input = tf.div(preprocessed_input, 127.0)

        red,green,blue=tf.split(preprocessed_input,num_or_size_splits=3,axis=3)
        preprocessed_input = (tf.multiply(blue, 0.2989) + tf.multiply(green, 0.5870) +
                              tf.multiply(red, 0.1140))
        return preprocessed_input

    def inference(self,input,reuse=False):
        preprocessed_inputs=self.pre_process(input)

        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(preprocessed_inputs, 32, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [3, 3],2, padding='SAME')   #[None,16,16,32]

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [3, 3],2, padding='SAME')    #[None,8,8,64]

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [3, 3],2, padding='SAME')    #[None,4,4,128]
            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
                net = tf.reduce_mean(net, axis=[1, 2], name="global_pool", keep_dims=True)     #[None,1,1,256]

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 28, [1, 1], activation_fn=None, padding='SAME',
                                               weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                               scope=scope, reuse=reuse)
            net = tf.contrib.layers.flatten(net)
            print("net_out:",net)

        return net

    def hard_triplet_loss(self,embeddings,lables,margin):
        triplet_hard_loss = self.triplet_loss_tool.batch_triplet_hard_loss(embeddings, lables, margin)
        return triplet_hard_loss


    def triplet_loss(self,ahchor_flatten,similar_flatten,dissimilar_flatten,lables,margin):
        triplet_loss,pos_dist,neg_dist = self.triplet_loss_tool.batch_triplet_loss(
             ahchor_flatten,similar_flatten,dissimilar_flatten,margin)
        return triplet_loss,pos_dist,neg_dist

    def loss(self,anchor_flatten,similar_flatten,dissimilar_flatten,lables,margin):
        triplet_loss, pos_dist, neg_dist = self.triplet_loss_tool.batch_triplet_loss(
            anchor_flatten, similar_flatten, dissimilar_flatten, margin)
        embeddings=tf.concat([anchor_flatten,similar_flatten,dissimilar_flatten],axis=0)

        hard_triplet_loss = self.triplet_loss_tool.batch_triplet_hard_loss(embeddings, lables, margin)
        return triplet_loss, pos_dist, neg_dist
