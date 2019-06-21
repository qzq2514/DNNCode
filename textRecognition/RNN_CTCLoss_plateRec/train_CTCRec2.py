import tensorflow as tf
import numpy as np
import time
import os
import cv2
from net import CTCRecognizer

from tensorflow.python.framework import graph_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_width=120
input_height=40

num_hidden = 64
batch_size = 64
snopshot = 200

char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","#"]

num_classes = 12
train_dir = "D:/forTensorflow/LprNet_CTCLoss_ISR/txt_labels/train"
test_dir = "D:/forTensorflow/LprNet_CTCLoss_ISR/txt_labels/test"

models_name="CTC_ISR2"
model_save_dir="models2/"

pb_path=os.path.join(model_save_dir,"pb")
ckpt_path=os.path.join(model_save_dir,"ckpt")

if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

def get_txt_paths(txt_dir):
    txt_paths=[]
    filenames=os.listdir(txt_dir)
    for file_name in filenames:
        txt_path=os.path.join(txt_dir,file_name)
        txt_paths.append(txt_path)
    return txt_paths

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

    batch_images_transpose = np.transpose(batch_images_data, axes=[0, 2, 1,3])
    batch_images_transpose=np.array(batch_images_transpose)

    sparse_batch_labels=get_sparse_labels(batch_labels)

    seq_len = np.ones(batch_size) * 120

    return batch_images_transpose,sparse_batch_labels,batch_labels,seq_len


def tf_read_iamges(txt_file_paths):
    txt_path = tf.train.string_input_producer(txt_file_paths)

    reader = tf.TextLineReader()
    _, value = reader.read(txt_path)
    img_filepath, label = tf.decode_csv(value, [[''], ['']], ' ')
    image_file = tf.read_file(img_filepath)

    input_shape=(input_height, input_width,3)
    rgb_image = tf.image.decode_jpeg(image_file, channels=3)

    red, green, blue = tf.split(rgb_image, num_or_size_splits=3, axis=2)
    bgr_image = tf.concat([blue ,green,red],axis=-1)

    # rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
    resized_imag = tf.image.resize_images(bgr_image, [input_height, input_width])
    resized_imag.set_shape(input_shape)

    # resized_image_tran=tf.transpose(resized_imag,[1,0,2])
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([resized_imag, label], batch_size=batch_size,
                                                      capacity=capacity,min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

#传入一个labels的集合,eg:["1234","837565",.....]
def get_sparse_labels(sequences):
   indices = []
   values = []
   for index, seq in enumerate(sequences):
      indices.extend(zip([index] * len(seq), range(len(seq))))
      seq=seq.decode('utf-8')
      values.extend(seq)

   indices = np.asarray(indices, dtype=np.int64)
   values = np.asarray(values, dtype=np.int64)
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
    train_txt_paths = get_txt_paths(train_dir)
    test_txt_paths = get_txt_paths(test_dir)

    #这里的train_label_batch中每个元素是字节类型(byte),需要在后面decode
    train_image_batch,train_label_batch=tf_read_iamges(train_txt_paths)
    test_image_batch,test_label_batch=tf_read_iamges(test_txt_paths)

    #train_image_batch:NHWC
    seq_len_placeholder = tf.fill([tf.shape(train_image_batch)[0]], tf.shape(train_image_batch)[2])
    ctcRecognizer = CTCRecognizer.CTCRecognizer(True, num_classes, num_hidden=num_hidden)
    processed_inputs=ctcRecognizer.preprocess(train_image_batch)
    logits = ctcRecognizer.inference(processed_inputs,seq_len_placeholder)

    decode_logits = ctcRecognizer.beam_searcn(logits, seq_len_placeholder)

    dense_predictions = tf.sparse_to_dense(decode_logits[0].indices,[tf.shape(train_image_batch,out_type=tf.int32)[0], input_width],
                                           decode_logits[0].values, default_value=-1,
                                           name='dense_predictions')

    sparse_gt = tf.py_func(get_sparse_labels, [train_label_batch], [tf.int64, tf.int64, tf.int64])
    sparse_groundtrouth = tf.to_int32(tf.SparseTensor(sparse_gt[0], sparse_gt[1], sparse_gt[2]))

    loss_mean = ctcRecognizer.loss(logits, sparse_groundtrouth, seq_len_placeholder)
    edit_distance_mean = ctcRecognizer.get_edit_distance_mean(decode_logits, sparse_groundtrouth)

    # 配置训练参数、学习率等
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 500, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss_mean, global_step=global_step)

    saver=tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())  #global_variables_initializer放在restore之前
        if os.path.exists(os.path.join(ckpt_path, "checkpoint")):
            print("restore--------------")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        while True:

            batch_loss, batch_edit_ditance, global_step_,learning_rate_, _ = sess.run([loss_mean, edit_distance_mean, global_step,learning_rate, train_step])
            info = "steps:{},batch_loss:{:.6f},edit_distance_mean:{:.6f},learning_rate:{:.6f}"
            print(info.format(global_step_, batch_loss, batch_edit_ditance, learning_rate_))

            if global_step_ % snopshot == 0:
                saver.save(sess,os.path.join(ckpt_path,models_name) , global_step=global_step_)
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense_predictions"])
                with tf.gfile.FastGFile(os.path.join(pb_path,models_name+"-" + str(global_step_) + ".pb"), mode="wb") as fw:
                   fw.write(constant_graph.SerializeToString())

                print("Successfully save model {}".format(models_name + str(global_step_)))

        coordinator.request_stop()
        coordinator.join(threads)
if __name__ == "__main__":
   train()