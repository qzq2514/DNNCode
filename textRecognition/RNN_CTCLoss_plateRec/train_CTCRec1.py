import tensorflow as tf
import numpy as np
import time
import os
import cv2
from net import CTCRecognizer

from tensorflow.python.framework import graph_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_width=120     #每个样本有多少个序列
input_height=40     #样本内每个序列的长度是多少

num_hidden = 64
batch_size = 64
snopshot = 200

#使用feed方式提供数据

char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","#"]
# 类别为10位数字+blank+ctc blank(即分为12类)
num_classes = 12
train_dir = "D:/forTensorflow/LprNet_CTCLoss_ISR/train"
test_dir = "D:/forTensorflow/LprNet_CTCLoss_ISR/test"

models_name="CTC_ISR1"
model_save_dir="models1/"

pb_path=os.path.join(model_save_dir,"pb")
ckpt_path=os.path.join(model_save_dir,"ckpt")

if not os.path.exists(pb_path):
    os.makedirs(pb_path)

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

#得到图片路径和对应的车牌号码字符串
def get_images_path(image_dir):
    image_paths=[]
    labels=[]

    filenames=os.listdir(image_dir)
    for file_name in filenames:
        label_str = file_name.split("_")[0]
        labels.append(label_str)
        image_path=os.path.join(image_dir,file_name)
        image_paths.append(image_path)
    return np.array(image_paths),np.array(labels)

def next_batch(is_random_sample,indices,image_paths,labels):
    if is_random_sample:
        indices=np.random.choice(len(image_paths),batch_size)
    elif indices==None:
        print("Please assign indices in the mode of random sampling!")
        return None,None
    try:
        batch_image_paths=image_paths[indices]
        batch_labels=labels[indices]
    except Exception as e:
        print("list index out of range while next_batch!")
        return None,None

    batch_images_data=[]
    for image_file_path in batch_image_paths:
        image=cv2.imread(image_file_path)

        image_resized=cv2.resize(image,(input_width,input_height))
        batch_images_data.append(image_resized)

    # batch_images_transpose = np.transpose(batch_images_data, axes=[0, 2, 1,3])
    batch_images_data=np.array(batch_images_data)
    sparse_batch_labels=get_sparse_labels(batch_labels)
    seq_len = np.ones(batch_size) * 120
    return batch_images_data,sparse_batch_labels,batch_labels,seq_len

def get_sparse_labels(sequences):
   indices = []
   values = []
   for index, seq in enumerate(sequences):
      indices.extend(zip([index] * len(seq), range(len(seq))))
      values.extend(seq)

   indices = np.asarray(indices, dtype=np.int64)
   values = np.asarray(values, dtype=np.int32)
   shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
   return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
   decoded_indexes = list()
   current_i = 0
   current_seq = []
   # print(sparse_tensor)
   for offset, i_and_index in enumerate(sparse_tensor[0]):  # sparse_tensor[0]是N*2的indices
      i = i_and_index[0]  # 一行是一个样本
      # print("i_and_index:",i_and_index)
      if i != current_i:  # current_is是当前样本的id
         decoded_indexes.append(current_seq)
         current_i = i
         current_seq = list()  # current_seq是当前样本预测值在sparse_tensor的values中对应的下标
      current_seq.append(offset)  # 之后通过下标就可以从sparse_tensor中找到对应的值
   decoded_indexes.append(current_seq)
   result = []
   for index in decoded_indexes:
      result.append(decode_a_seq(index, sparse_tensor))
   return result


def decode_a_seq(indexes, spars_tensor):
   decoded = []
   for m in indexes:
      # print("m:",m)
      str_id = spars_tensor[1][m]
      # print(m, "---", str_id)
      str = char_set[str_id]
      decoded.append(str)
   return decoded

def get_accuarcy1(decode_predictions_, no_parse_labels):
    total_num=len(no_parse_labels)
    correct_num=0
    for ind,val in enumerate(decode_predictions_):
        pred_number = ''
        for code in val:
            pred_number += char_set[code]
        pred_number=pred_number.strip("#")
        is_correct=pred_number==no_parse_labels[ind]
        if is_correct:
            correct_num+=1
        print("{}:{}------>{}".format(is_correct,no_parse_labels[ind],pred_number))
    return correct_num/total_num

def get_accuarcy2(decoded_logits, sparse_labels):
    try:
        sparse_labels_list = decode_sparse_tensor(sparse_labels)
        decoded_list = decode_sparse_tensor(decoded_logits)
        true_numer = 0

        if len(decoded_list) != len(sparse_labels_list):
          print("No match :{}--->{}".format(len(sparse_labels_list),len(decoded_list)))
          return None  # edit_distance起作用

        for idx, pred_number in enumerate(decoded_list):
          groundTruth_number = sparse_labels_list[idx]
          cur_correct = (pred_number == groundTruth_number)
          info_str = "{}:{}-({}) <-------> {}-({})". \
             format(cur_correct, groundTruth_number, len(groundTruth_number), pred_number, len(pred_number))
          print(info_str)
          if cur_correct:
             true_numer = true_numer + 1
        accuary = true_numer * 1.0 / len(decoded_list)
        return accuary
    except Exception as e:
        print("Exception in get_accuarcy()!!")
        return None

# 定义训练过程
def train():
    train_image_paths, train_labels = get_images_path(train_dir)
    test_image_paths, test_labels = get_images_path(test_dir)

    inputs = tf.placeholder(tf.float32, [None, input_height,input_width,3],name="inputs")
    sparse_groundtrouth = tf.sparse_placeholder(tf.int32,name="sparse_gt")
    seq_len_placeholder = tf.placeholder(tf.int32, [None],name="seq_len_gt")

    ctcRecognizer = CTCRecognizer.CTCRecognizer(True, num_classes, num_hidden=num_hidden)
    processed_inputs=ctcRecognizer.preprocess(inputs)
    logits = ctcRecognizer.inference(processed_inputs,seq_len_placeholder)
    decode_logits = ctcRecognizer.beam_searcn(logits, seq_len_placeholder)

    dense_predictions = tf.sparse_to_dense(decode_logits[0].indices,[tf.shape(inputs,out_type=tf.int32)[0], 120],
                                           decode_logits[0].values, default_value=-1,
                                           name='dense_predictions')

    loss_mean=ctcRecognizer.loss(logits,sparse_groundtrouth,seq_len_placeholder)
    edit_distance_mean = ctcRecognizer.get_edit_distance_mean(decode_logits, sparse_groundtrouth)

    # 配置训练参数、学习率等
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 1000, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step=optimizer.minimize(loss_mean, global_step=global_step)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(os.path.join(model_save_dir,"ckpt","checkpoint")):
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(model_save_dir,"ckpt")))

        while True:
            train_inputs, train_targets,no_parse_labels, train_seq_len = next_batch(True, None, train_image_paths, train_labels)

            train_feed = {inputs: train_inputs, sparse_groundtrouth: train_targets,seq_len_placeholder: train_seq_len}

            batch_loss, batch_edit_ditance, global_step_, learning_rate_,_ = sess.run([loss_mean, edit_distance_mean, global_step,learning_rate, train_step],feed_dict=train_feed)

            info = "steps:{},batch_loss:{:.6f},edit_distance_mean:{:.6f},learning_rate:{:.6f}"

            print(info.format(global_step_, batch_loss,batch_edit_ditance,learning_rate_))

            if global_step_ % snopshot == 0:

                test_inputs, test_targets,no_parse_labels, test_seq_len = next_batch(True, None, test_image_paths, test_labels)
                test_feed = {inputs: test_inputs,seq_len_placeholder: test_seq_len}
                decode_predictions_,test_decode_logits = sess.run([dense_predictions,decode_logits[0]], feed_dict=test_feed)

                test_batch_acc1 = get_accuarcy1(decode_predictions_, no_parse_labels) #非稀疏矩阵形式
                test_batch_acc2 = get_accuarcy2(test_decode_logits, test_targets)     #稀疏矩阵形式
                saver.save(sess,os.path.join(ckpt_path,models_name) , global_step=global_step_)

                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense_predictions"])
                with tf.gfile.FastGFile(os.path.join(pb_path,models_name+"-" + str(global_step_) + ".pb"), mode="wb") as fw:
                    fw.write(constant_graph.SerializeToString())
                print("目前准确率1:{},准确率2:{}".format(test_batch_acc1,test_batch_acc2))


if __name__ == "__main__":
   train()