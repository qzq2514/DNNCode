import tensorflow as tf
import numpy as np
import os
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_width=120
input_height=40

num_hidden = 64
batch_size = 1

char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","#"]
eval_dir = "D:/forTensorflow/LprNet_CTCLoss_ISR/train"
model_path = "models1/ckpt/CTC_ISR1-4800"
num_classes = 12

def ckeck_correct(decode_predictions_, no_parse_labels):
    is_correct=False
    for ind, val in enumerate(decode_predictions_):
        pred_number = ''
        for code in val:
            pred_number += char_set[code]
        pred_number = pred_number.strip("#")
        is_correct = pred_number == no_parse_labels[ind]
        print("{}------->{}".format(no_parse_labels[ind], pred_number))
    return is_correct

def train():

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        seq_len_placeholder = tf.get_default_graph().get_tensor_by_name('seq_len_gt:0')
        dense_predictions = tf.get_default_graph().get_tensor_by_name('dense_predictions:0')

        correct_num=0
        for total_num, filename in enumerate(os.listdir(eval_dir)):
            labels_batch = []
            file_path = os.path.join(eval_dir, filename)
            label = os.path.basename(file_path).split("_")[0]
            org_color_image = cv2.imread(file_path)
            color_image_resized = cv2.resize(org_color_image, (input_width, input_height))

            # color_image_tran=np.transpose(color_image_resized,axes=[1,0,2])
            color_image_tran_batch=color_image_resized[np.newaxis,:]
            labels_batch.append(label)
            seq_len = np.ones(batch_size) * 120

            eval_dict={inputs: color_image_tran_batch,seq_len_placeholder: seq_len}

            dense_predictions_ = sess.run(dense_predictions, feed_dict=eval_dict)

            print(dense_predictions_.shape)
            is_correct = ckeck_correct(dense_predictions_, labels_batch)
            correct_num = correct_num + 1 if is_correct else correct_num

            print("accuarcy:{}".format(correct_num / (total_num + 1)))
            # cv2.imshow("org_color_image", org_color_image)
            # cv2.waitKey(0)


if __name__ == "__main__":
   train()