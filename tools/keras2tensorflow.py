import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

import sys
from keras import backend as K
from keras.models import load_model
from tensorflow.python.framework import graph_util,graph_io


input_h5_path=sys.argv[1]
    # "P:/WorkSpace/myGithub/DNNCode/landmark/plateCorner/" \
    #           "LprLocNet/models/plateCornerDetFull30.h5"

output_pb_path=input_h5_path[:-2]+"pb"

def h5_2_pb(h5_model):

    #打印原模型的输入和输出节点
    print("inputs:")
    for input_node in h5_model.inputs:
        print(input_node, "----->", "input")
        # 即便这里对输入节点进行identity,但是在使用pb预测是仍无法通过"input"节点名找到对应的节点
        # 所以还是使用原来的节点名称
        # tf.identity(input_node, "input")

    print("------\noutputs:")
    for output_node in h5_model.outputs:
        # 可根据实际情况将对应的模型原输出映射至对应的名称
        print(output_node,"----->","landmark_output")
        tf.identity(output_node,"landmark_output")

    #保存成pb模型
    sess=K.get_session()
    init_graph = sess.graph.as_graph_def()
    const_graph=graph_util.convert_variables_to_constants(sess,init_graph,["landmark_output"])
    with tf.gfile.FastGFile(output_pb_path,"wb") as fw:
        fw.write(const_graph.SerializeToString())

    #也可以通过该种方式保存pb模型
    # graph_io.write_graph(const_graph,"path/to/save/dir/",name = "save/pb/name",as_text = False)

if __name__=="__main__":
    #load_model只能加载同时保存网络架构和权重参数的模型,即在训练时通过save保存的模型,而非通过save_weights保存的模型
    h5_model=load_model(input_h5_path)
    h5_2_pb(h5_model)

    print("Successfullt convert h5 model to pb model!!")


