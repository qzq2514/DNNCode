import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_io
from nets import WideResNet

#将模型中batch norm层的is_training置为False,并且去掉dropout层
#其实主要是为了去掉dropout层,因为is_training在训练时默认placeholder为False
input_width=28
input_height=28
class_num=36
checkpoint_path ="models/ckpt/WideResNet.ckpt-1000"
frozen_path ="models/pb/WideResNet_frozen.pb"

def freezing_graph(checkpoint, frozen_path):
  graph = tf.Graph()
  with graph.as_default():
    # 正好也可以通过此方式将原模型中所有的is_training置为False
    # 这里主要是将slim.dropout中的is_training置为False,因为在构建模型的时候
    # 仅仅slim.batch_norm是使用is_training参数的(以后可以使用方式二构建参数空间,
    # 详见WideResNet.py第119行之后)
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):

      input_tensor = tf.placeholder(tf.float32, [None,input_height, input_width ,3],name="inputs")

      wideResNet=WideResNet.WideResNet(is_training=False , num_classes=class_num,
                                     keep_prob=1.0,unit_num_in_block=12,width_k = 2)

      processed_inputs = wideResNet.preprocess(input_tensor)
      logits = wideResNet.inference(processed_inputs)

      softmax_output, classes = wideResNet.postprocess(logits)
      softmax_output_ = tf.identity(softmax_output, name="softmax_output")

      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  sess = tf.Session(graph=graph)
  sess.run(init)
  saver.restore(sess, checkpoint)

  constant_graph =  tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax_output"])
  with tf.gfile.FastGFile(frozen_path, mode="wb") as fw:
      fw.write(constant_graph.SerializeToString())

def main():

  freezing_graph(checkpoint_path, frozen_path)
  print("Successfully freeze ckpt to pb!")

if __name__=="__main__":
  main()