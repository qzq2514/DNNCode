import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim = tf.contrib.slim    #最好用这种形式

#subsample只作用于跳跃连接的identity mapping,
#而主路则就通过按部就班的指定卷积和步长进行即可
def subsample(inputs,factor,scope):
    if factor==1:   #此时identity mapping处于某个residual block的非最后一个residual unit，不需要下采样
        return inputs #
    else:           #此时identity mapping处于某个residual block的最后一个residual unit，需要采用下采样(H,W均变一半)
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,out_channals,kernel_size,stride,scope=None):
    if stride==1:
        return slim.conv2d(inputs,out_channals,kernel_size,stride=1,
                           padding="SAME",scope=scope+"_same")

    else:
        #有点没搞懂这里填充的意义,既然是为了下采样,直接步长为2的"SAME"卷积不行？
        #方法一
        # pad_total=kernel_size-1
        # pad_beg=pad_total//2
        # pad_end=pad_total-pad_beg
        #
        # inputs=tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        # return slim.conv2d(inputs, out_channals, kernel_size, stride=stride, padding="VALID", scope=scope)

        # 方法2
        # 好像是可以的,直接步长为2的"SAME"卷积
        # 因为feature map的宽高Hin、Win都是2的整倍数,且stride=2
        # 如果是方法一,则输出的宽为Hout=math.floor((Hin-1)/stride+1)=Hin/2
        # 如果是方法一,则输出的宽为Hout=math.floor(Hin/stride)=Hin/2,两者相等
        return slim.conv2d(inputs,out_channals,kernel_size,
                           stride=stride,padding="SAME",scope=scope+"_same")

#forward in a residual unit
@slim.add_arg_scope
def ResUnit_forward(inputs,unit_output_channals,top_2_filter_nums,mid_conv_stride):
    # 已通过slim.arg_scope设置:
    # 1. slim.arg_scope进行设置-slim.conv2d的activation_fn=None,normalizer_fn都是None
    # 2. slim.batch_norm的activation_fn=tf.nn.relu
    input_depth=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)  #获得输入通道数

    # resNetV2核心1:前置激活-将bn和激活函数放在卷积层之前
    # 对主路残差部分进行前向传播
    preActivation=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope="preActivation")

    if input_depth==unit_output_channals:
        shortcut=subsample(inputs,mid_conv_stride,scope="shortcut")
    else:
        shortcut=slim.conv2d(preActivation,unit_output_channals,[1,1],
                             normalizer_fn=None, activation_fn=None,
                             stride=mid_conv_stride,scope="shortcut")

    # 按理说前置bn和激活,这里的conv2d需要将normalizer_fn置为空
    # 但是这里要为下一次的conv进行bn和前置激活(通过slim.arg_scope已经设置slim.conv2d有bn和relu了)
    residual = slim.conv2d(preActivation, top_2_filter_nums, [1,1],stride=1, scope="conv1")
    residual = conv2d_same(residual, top_2_filter_nums,3, stride=mid_conv_stride, scope="conv2")
    #残差分支最后一个卷积不要做bn和relu,因为是前置激活,该卷积已经在之前做了bn和relu
    residual = slim.conv2d(residual, unit_output_channals, [1, 1],
                           normalizer_fn=None, activation_fn=None,
                           stride=1,scope="conv3")

    # resNetV2核心2:在前置激活后,残差单元在进行逐像素相加后不要再激活
    unit_output=shortcut+residual

    return unit_output

#parse residual block and stack blocks
@slim.add_arg_scope
def stack_ResBlocks(net,blocks):
    for block in blocks:
        with tf.variable_scope(block.scope,"ResBlock",[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope("unit_/%d"%(i+1),values=[net]):
                    #Block is like [(256, 64, 1)] * 2 + [(256, 64, 2)]
                    unit_output_channals,top_2_filter_nums,mid_conv_stride=unit
                    net=block.unit_fn(net,unit_output_channals=unit_output_channals,
                                      top_2_filter_nums=top_2_filter_nums,
                                      mid_conv_stride=mid_conv_stride)
    return net
