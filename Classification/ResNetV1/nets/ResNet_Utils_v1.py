import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim = tf.contrib.slim    #最好用这种形式



# class ResBlock(collections.namedtuple("Block",["scope","unit_fn","args"])):
#     'A named tuple describing a residual block'
#     'a residual block has many residual units and ' \
#     'a residual unit consist there conv layers'
#     'eg..Block("block1", bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])'
#     '每个residual block会在最后一个residual unit的最后一个卷积下做步长为2的卷积，达到下采样(H,W均变一半)的效果'

#subsample只作用于跳跃连接的identity mapping,
#而主路则就通过按部就班的指定卷积和步长进行即可
def subsample(inputs,factor,scope):
    if factor==1:   #此时identity mapping处于某个residual block的非最后一个residual unit，不需要下采样
        return inputs #
    else:           #此时identity mapping处于某个residual block的最后一个residual unit，需要采用下采样(H,W均变一半)
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,out_channals,kernel_size,stride,scope=None):
    if stride==1:
        return slim.conv2d(inputs,out_channals,kernel_size,stride=1,padding="SAME",scope="scope")

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
        return slim.conv2d(inputs,out_channals,kernel_size,stride=stride,padding="SAME",scope=scope)

#forward in a residual unit
@slim.add_arg_scope
def ResUnit_forward(inputs,unit_output_channals,top_2_filter_nums,mid_conv_stride):
    input_depth=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)  #获得输入通道数

    # 对跳跃连接的全等映射(identity mapping)做处理
    # 不仅要保证identity mapping后的通道数和该unit的输出通道数(unit_output_channals)相同,
    # 还要保证在下采样的时候,identity mapping宽高都减半

    # 在ResNet-50/101/200结构中只有在每个Block的最后一个unit才会进行下采样,此时stride=2
    # 对shortcut处理:只需要保证shortcut和指定的单元输出通道数相同,
    # 其余只需要把步长直接传给下面的函数(subsample,slim.conv2d)就行
    if input_depth==unit_output_channals:#输入通道数和输出通道数相同,则可以直接做残差
        shortcut=subsample(inputs,mid_conv_stride,scope="shortcut")
    else:  #如果通道数不一样，则需要通过1x1卷积,并指定卷积核个数保证通道数一致,这样才能做和主路的残差结构进行按位的求和运算
           #同时,这里还通过指定stride=mid_conv_stride，保证无论当前unit是不是所在block的最后一个unit,
           #都能够达到下采样(是最后一个,mid_conv_stride=2)或者非下采样的效果(mid_conv_stride=1)
        shortcut=slim.conv2d(inputs,unit_output_channals,[1,1],stride=mid_conv_stride,
                             activation_fn=None,scope="shortcut")

    #对主路残差部分进行前向传播
    residual = slim.conv2d(inputs, top_2_filter_nums, [1,1], stride=1, scope="conv1")
    residual = conv2d_same(residual, top_2_filter_nums,3, stride=mid_conv_stride, scope="conv2")
    residual = slim.conv2d(residual, unit_output_channals, [1, 1], stride=1,
                            activation_fn=None,scope="conv3")

    #对identity mapping和残差部分进行按位求和(ResNet的核心)再激活
    unit_output=tf.nn.relu(shortcut+residual)
    return unit_output

#parse residual block and stack blocks
@slim.add_arg_scope
def stack_ResBlocks(net,blocks):
    for block in blocks:
        with tf.variable_scope(block.scope,"ResBlock",[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope("unit_/%d"%(i+1),values=[net]):
                    #unit is like [(256, 64, 1)] * 2 + [(256, 64, 2)]
                    unit_output_channals,top_2_filter_nums,mid_conv_stride=unit
                    net=block.unit_fn(net,unit_output_channals=unit_output_channals,
                                      top_2_filter_nums=top_2_filter_nums,
                                      mid_conv_stride=mid_conv_stride)
    return net
