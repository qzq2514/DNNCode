import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DenseNetForDigitmy(object):
    def __init__(self, is_training, num_classes,growth_rate,net_depth,compression):
        self.num_classes = num_classes
        self._is_training = is_training
        self.growth_rate=growth_rate         #在DenseNet Block内的卷积层的宽度(通道数)
        self.conv_num=int((net_depth-4)/3)   #共3个DenseNet Block,每个DenseNet Block内的卷积次数
                                             #(去掉开始的一次卷积和结尾的bn、全局池化和全连接)
        self.compression=compression


    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        preprocessed_inputs = tf.tzv(preprocessed_inputs, 128.0)
        return preprocessed_inputs

    #在每个block内的每一层采用密集连接,即某层的输入是在该层之前且在该block内的所有层的输出的合并
    #不包括其他blcok内任意一层的输出,仅仅与当前block内的层有关
    #(bottleneck中有1x1卷积的称为DenseNet-B)
    def bottleneck(self,inputs,name):
        with tf.variable_scope(name) as sc:
            cur_bn1 = slim.batch_norm(inputs, scope="bn1")
            #该1x1卷积是bottleneck的核心,为了防止在一个block中后面层的层因为密集连接导致其输入特征图通道数太大
            #这里通过1x1卷积将其固定到一定的尺寸(原文中是4倍的self.growth_rate)
            cur_conv1x1 = slim.convolution2d(cur_bn1, num_outputs=4*self.growth_rate,
                                  kernel_size=1, stride=1)
            cur_bn2 = slim.batch_norm(cur_conv1x1, scope="bn2")
            cur_conv3x3 = slim.convolution2d(cur_bn2, num_outputs=self.growth_rate,
                                          kernel_size=3, stride=1)
            return cur_conv3x3

    #在一个block中将前面层的输出不断合并作为后面层的输入
    def add_composite(self,cur_block_layer_collection,name):
        cur_bottleneck=self.bottleneck(cur_block_layer_collection,name)
        cur_block_layer_collection=tf.concat([cur_bottleneck,cur_block_layer_collection],axis=-1)
        return cur_block_layer_collection

    # transition中在下采样的步长为2的平均池化之前有一个1x1卷积,该1x1卷积再压缩系数的控制下将上一个的Dense block
    # 的输出,将其压缩到一半的通道数(此时网络成为DenseNet-C)
    def transition(self,block_output,name):
        block_output_channel=block_output.get_shape().as_list()[-1]
        with tf.variable_scope(name) as sc:
            cur_bn=slim.batch_norm(block_output,scope="bn")

            cur_conv=slim.convolution2d(cur_bn,num_outputs=int(block_output_channel*self.compression),
                                        activation_fn=tf.nn.relu,kernel_size=1,stride=1,scope="conv")
            cuv_pool=slim.avg_pool2d(cur_conv,kernel_size=2,stride=2,scope="pool")
        return cuv_pool

    #inputs:[batch_size,28,28,3]
    def inference(self, inputs):
        print("Using DenseNet L=40,K=12.....")
        with slim.arg_scope(self.DenseNet_arg_scope(is_training=self._is_training)):
            pre_conv = slim.convolution2d(inputs,num_outputs=16,kernel_size=3,stride=1,padding="SAME")

            with tf.variable_scope("block1") as sc:
                cur_block1_layer_collection=self.bottleneck(pre_conv,"layer1")
                for conv_id in range(self.conv_num-1):
                    cur_block1_layer_collection=self.add_composite(cur_block1_layer_collection,
                                                                   "layer" + str(conv_id+2))

                #某个transition层的输入仅仅是上一个block的输出,并不与之前其他的block输出合并
                transition_layer1=self.transition(cur_block1_layer_collection,"transition")

            with tf.variable_scope("block2") as sc:
                cur_block2_layer_collection = self.bottleneck(transition_layer1, "layer1")
                for conv_id in range(self.conv_num - 1):
                    cur_block2_layer_collection = self.add_composite(cur_block2_layer_collection,
                                                                     "layer" + str(conv_id + 2))
                transition_layer2 = self.transition(cur_block2_layer_collection, "transition")
            #最后一个block后面不用添加变换层
            with tf.variable_scope("block3") as sc:
                cur_block3_layer_collection = self.bottleneck(transition_layer2, "layer1")
                for conv_id in range(self.conv_num - 1):
                    cur_block3_layer_collection = self.add_composite(cur_block3_layer_collection,
                                                                    "layer" + str(conv_id + 2))
                # transition_layer3 = self.transition(cur_block3_layer_collection, "transition")

            post_conv=slim.convolution2d(cur_block3_layer_collection,num_outputs=self.growth_rate*2,kernel_size=3,
                                                stride=2,activation_fn=tf.nn.relu,scope="transition_conv")
            post_conv_bn=slim.batch_norm(post_conv,scope="bn_last")

            net_global_pool = tf.reduce_mean(post_conv_bn, [1, 2],name="global_pool",keep_dims=True)

            net = slim.convolution2d(net_global_pool, num_outputs=self.num_classes,
                                     kernel_size=1, activation_fn=None, normalizer_fn=None, scope="full_conv")

            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            print("logits:",logits)
        return logits

    def postprocess(self,logits):
        softmax=tf.nn.softmax(logits)
        classes=tf.cast(tf.argmax(softmax,axis=1),tf.int32)
        return softmax,classes

    def loss(self,logits,labels):
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        tf.add_to_collection("Loss",softmax_loss)
        loss_all=tf.add_n(tf.get_collection("Loss"),name="total_loss")
        return loss_all

    def DenseNet_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.90,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            "activation_fn":tf.nn.relu
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        #DenseNet借鉴resNetV2,采用前置激活,不在卷积后进行bn和relu
        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=None):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.avg_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc