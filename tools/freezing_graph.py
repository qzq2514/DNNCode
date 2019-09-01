import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_io
from net import CTCRecognizer

input_width=120
input_height=40
class_num=12
num_hidden=64
checkpoint_path ="models2/ckpt/CTC_ISR2-13000"
frozen_path ="freezeModel/CTC_ISR_frezee.pb"

def freezing_graph(checkpoint, frozen_path):
  graph = tf.Graph()
  with graph.as_default():
    # 正好也可以通过此方式将原模型中所有的is_training置为False
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):

      input_tensor = tf.placeholder(tf.float32, [None,input_height, input_width ,3],name="inputs")
      seq_len_placeholder = tf.fill([tf.shape(input_tensor)[0]], tf.shape(input_tensor)[2])

      ctcRecognizer=CTCRecognizer.CTCRecognizer(False, class_num, num_hidden=num_hidden)
      processed_inputs = ctcRecognizer.preprocess(input_tensor)
      logits = ctcRecognizer.inference(processed_inputs, seq_len_placeholder)
      #logits:(max_step,batch_size,num_class)

      data_length = tf.fill([tf.shape(logits)[1]], tf.shape(logits)[0])
      result = tf.nn.ctc_greedy_decoder(logits, data_length, merge_repeated=True)
      predictions = tf.to_int32(result[0][0])
      tf.sparse_to_dense(predictions.indices, [tf.shape(input_tensor, out_type=tf.int64)[0],
                                               tf.shape(input_tensor, out_type=tf.int64)[2]],
                         predictions.values, default_value=-1, name='dense_predictions')
      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  sess = tf.Session(graph=graph)
  sess.run(init)
  saver.restore(sess, checkpoint)

  constant_graph =  tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense_predictions"])
  with tf.gfile.FastGFile(frozen_path, mode="wb") as fw:
      fw.write(constant_graph.SerializeToString())

def main():

  freezing_graph(checkpoint_path, frozen_path)
  print("Successfully freeze ckpt to pb!")

if __name__=="__main__":
  main()