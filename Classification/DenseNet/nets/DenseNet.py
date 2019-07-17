import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DenseNet(object):
    def __init__(self, is_training, num_classes,growth_rate,net_depth):
        self.num_classes = num_classes
        self._is_training = is_training
        self.growth_rate=growth_rate         #在DenseNet Block内的卷积层的宽度(通道数)
        self.conv_num=int((net_depth-4)/3)   #共3个DenseNet Block,每个DenseNet Block内的卷积次数
                                             #(去掉开始的一次卷积和结尾的bn、全局池化和全连接)


    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    # 每经过一个卷积操作,均把卷积结果和未卷积之前的集合进行合(论文的核心)
    # 以供后面的卷积能够直接连接到之前的卷积结果,起到Densely connection的效果
    # 其中原文中在Dense Unit的3x3卷积之前还有一个Bottleneck层,
    # 会将目前为止该Dense block的合并后的特征图固定到4xgrowth_rate(此时网络称为DenseNet-B)再送入3x3卷积
    # 这样保证不至于依次将之前的feature map合并后产生的特征图通道数过多
    # 同时还有一个压缩层,在该后通过1x1卷积再将该Dense Block的特征图的通道数减少一定的倍数(此时网络称为DenseNet-C)
    # 4xgrowth_rate的过渡层和压缩层都有的叫做DenseNet-BC
    # 这里复现仅仅复现了DenseNet的核心:密集连接,去掉了Bottleneck层压缩层,这样在复现的难度上大大减少:无论在哪个Dense Block
    # 的哪一层，只需要将之间所有Dense Block的所有层全部添加进去卷积就行
    def add_layer(self,layer_collection,name):
        #根据论文第三章的Composite function的一节,在本层依次进行bn,relu和3x3conv
        #将bn和relu放在conv之前也符合ResNetV2中前置激活的想法,保证在合并(tf.concat)之后
        #不再进行bn和激活
        with tf.variable_scope(name) as sc:
            cur_bn=slim.batch_norm(layer_collection,scope="bn")
            cur_conv=slim.convolution2d(cur_bn,num_outputs=self.growth_rate,
                                        kernel_size=3,stride=1)
            layer_collection=tf.concat([cur_conv,layer_collection],axis=-1)
        return layer_collection

    # 每个DenseNet Block之间的转换层,仅仅用于下采样,且最后不会添加到layer_collection中
    def add_transition(self,layer_collection,name):
        layer_collection_channel=layer_collection.get_shape().as_list()[-1]
        with tf.variable_scope(name) as sc:
            cur_bn=slim.batch_norm(layer_collection,scope="bn")

            # 这里我们transition层中的1x1卷积不会改变通道数,注意在conv和pool之间要加一个relu,
            # 因为在参数空间内,convolution2d是被设置为不跟激活函数的
            cur_conv=slim.convolution2d(cur_bn,num_outputs=layer_collection_channel,activation_fn=tf.nn.relu,
                                        kernel_size=1,stride=1,scope="conv")
            cuv_pool=slim.avg_pool2d(cur_conv,kernel_size=2,stride=2,scope="pool")
        return cuv_pool

    #inputs:[batch_size,32,32,3]
    def inference(self, inputs):
        with slim.arg_scope(self.DenseNet_arg_scope(is_training=self._is_training)):
            layer_collection = slim.convolution2d(inputs,num_outputs=16,kernel_size=3,stride=1,padding="SAME")

            with tf.variable_scope("block1") as sc:
                for conv_id in range(self.conv_num):
                    layer_collection=self.add_layer(layer_collection, "layer" + str(conv_id))
                layer_collection=self.add_transition(layer_collection,"transition")

            print("layer_collection:", layer_collection)
            with tf.variable_scope("block2") as sc:
                for conv_id in range(self.conv_num):
                    layer_collection=self.add_layer(layer_collection, "layer" + str(conv_id))
                layer_collection=self.add_transition(layer_collection,"transition")

            print("layer_collection:", layer_collection)
            #最后一个block后面不用添加变换层
            with tf.variable_scope("block3") as sc:
                for conv_id in range(self.conv_num):
                    layer_collection=self.add_layer(layer_collection, "layer" + str(conv_id))

            print("layer_collection:",layer_collection)
            layer_collection=slim.batch_norm(layer_collection,scope="bn_last")

            net_global_pool = tf.reduce_mean(layer_collection, [1, 2],name="global_pool",keep_dims=True)

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