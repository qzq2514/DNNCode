import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.ResNet_Utils_v1 import ResUnit_forward,stack_ResBlocks,conv2d_same

class Block(collections.namedtuple("Block",["scope","unit_fn","args"])):
    'A named tuple describing a residual block'
    '一个残差 block 有多个残差 units' \
    '一个残差 unit 包括三个卷积操作'
    '如Block("block1", ResUnit_forward, [(256, 64, 1)] * 2 + [(256, 64, 2)])'
    '其中一个残差unit用一个三个元素的list表示,eg:(256, 64, 1)如下解释' \
    '其中 256 : 当前unit最后一个卷积的卷积核个数，也就是整个unit输出的feature map的通道数'
    '64:前两层卷积的卷积核个数' \
    '1 : 中间一层卷积操作的步长(PS:其余两个卷积操作步长均为1)'

class ResNetV1(object):
    def __init__(self, is_training, num_classes,global_pool=True,
                 indlude_root_block=True,resNet_type="resNet_Digit"):
        self.num_classes = num_classes
        self._is_training = is_training
        self.global_pool=global_pool
        self.indlude_root_block=indlude_root_block
        self.resNet_type=resNet_type

    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def resNet_Digit(self):
        resNet_Digit = [
            Block("block1", ResUnit_forward, [(256, 128, 1)] * 2 + [(256, 64, 2)]),
            # Block("block2", ResUnit_forward, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            # Block("block3", ResUnit_forward, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            Block("block4", ResUnit_forward, [(1024, 512, 1)] * 3 + [(512, 128, 2)]),
        ]
        return resNet_Digit

    def resNet_50(self):
        resNet_50_blocks=[
            Block("block1", ResUnit_forward, [(256, 64,  1)] * 2 + [(256, 64,  2)]),
            Block("block2", ResUnit_forward, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block("block3", ResUnit_forward, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            Block("block4", ResUnit_forward, [(2048, 512, 1)] * 3 + [(512, 128, 2)]),]
        return resNet_50_blocks

    def resNet_101(self):
        resNet_101_blocks = [
            Block('block1', ResUnit_forward, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block('block2', ResUnit_forward, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            Block('block3', ResUnit_forward, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
            Block('block4', ResUnit_forward, [(2048, 512, 1)] * 3)
        ]
        return resNet_101_blocks


    def resNet_152(self):
        resNet_152_blocks = [
            Block( 'block1', ResUnit_forward, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block('block2', ResUnit_forward, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
            Block('block3', ResUnit_forward, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            Block('block4', ResUnit_forward, [(2048, 512, 1)] * 3)]
        return resNet_152_blocks

    def resNet_200(inputs):
        resNet_200_blocks = [
            Block('block1', ResUnit_forward, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            Block('block2', ResUnit_forward, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
            Block('block3', ResUnit_forward, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            Block('block4', ResUnit_forward, [(2048, 512, 1)] * 3)]
        return resNet_200_blocks

    def resNet_inference(self,inputs,blocks,scope=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
            net = inputs
            if self.indlude_root_block:  #从残差卷积前的前两个步骤开始inference
                net=conv2d_same(net,64,7,stride=2,scope="conv1")
                net=slim.max_pool2d(net,[3,3],stride=2,scope="pool1")
            net=stack_ResBlocks(net,blocks)
            print("net_shape:",net)
            if self.global_pool:
                # Global average pooling.
                # 全局平均池化,使用tf.reduce_mean比使用tf.avg_pool更快
                # 这里指定要平均的维度是宽高维度,即[1,2]维度。
                # keep_dims用于保持维度,原是[N,H,W,C],现在H和W维度平均后变成[N,1,1,C]
                net=tf.reduce_mean(net,[1,2],name="pool5",keep_dims=True)
            # 最后使用1x1卷积将feature map变成[N,1,1,num_classes]大小
            net=slim.conv2d(net,self.num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope="full_conv")
            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        return logits

    # 宽度乘数在(0,1]之间,改变卷积核个数，进而改变输入和输出的通道数
    def inference(self, inputs):
        if self.resNet_type=="resNet_Digit":
            resNet_Blocks=self.resNet_Digit()
        elif self.resNet_type=="resNet_50":
            resNet_Blocks=self.resNet_50()
        elif self.resNet_type=="resNet_101":
            resNet_Blocks = self.resNet_101()
        elif self.resNet_type == "resNet_152":
            resNet_Blocks = self.resNet_152()
        elif self.resNet_type == "resNet_200":
            resNet_Blocks = self.resNet_200()
        else:
            resNet_Blocks = self.resNet_50()  #默认
            self.resNet_type="resNet_50"
            print("Wrong resNet version,using resNet_50!")
        scope = self.resNet_type
        print("using:{}".format(self.resNet_type))

        with slim.arg_scope(self.ResNet_arg_scope(self._is_training)):
            logits=self.resNet_inference(inputs,resNet_Blocks,scope=scope)
        return logits

    def postprocess(self,logits):
        softmax=tf.nn.softmax(logits)
        classes=tf.cast(tf.argmax(softmax,axis=1),tf.int32)
        #softmax:  N*num_classes  ,classes:N*1
        #其中N为样本数
        return softmax,classes

    def loss(self,logits,labels):
        # tf.nn.sparse_softmax_cross_entropy_with_logits将元素值在[0,num_classes-1]范围的标签自动转为ont-hot形式，
        # 然后计算每个样本的损失,返回一个长度为N的向量,其中N为样本数
        # 交叉熵损失有log运算,为了防止出现0,则加一个微笑的正数(1e-8)
        # 其中logits是神经网络输出层的结果，而非softmax后的结果
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        tf.add_to_collection("Loss",softmax_loss)
        loss_all=tf.add_n(tf.get_collection("Loss"),name="total_loss")
        return loss_all

    def ResNet_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):
        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc