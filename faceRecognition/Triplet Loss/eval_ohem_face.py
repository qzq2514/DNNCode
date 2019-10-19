import numpy as np
import tensorflow as tf
import json
import time
import cv2
import os
import dataset

model_path = 'models/pb/siamese_triplet_28out_bn_face2-200.pb'
eval_root =   "P:/WorkSpace/VS2015/FaceRecognition/FaceRecognition/trainData/faceData/dataBase41"
# eval_root = "D:/forTensorflow/SimpleLFW/LFW_test"

batch_size=16
input_height=32
input_width=32
channals=3
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]

sess=tf.Session()
with tf.gfile.FastGFile(model_path, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sess.run(tf.global_variables_initializer())

input = sess.graph.get_tensor_by_name('anchor:0')
output = sess.graph.get_tensor_by_name('flatten_anchor:0')

def get_base_feature(test_dataset):
    dir="D:/forTensorflow/charRecTrain/forMyDNNCode/base"
    base_features = {}
    for image_name in os.listdir(dir):
        image_path = os.path.join(dir,image_name)
        label = image_name[0]
        image_data=test_dataset.process_data(image_path)[np.newaxis, :]
        cur_feature = sess.run(output, feed_dict={input: image_data})
        base_features[label[0]] = cur_feature[0]
    return base_features

def get_random_base_feature(test_dataset):
    # numbers = list(map(chr, range(ord('0'), ord('9') + 1)))
    # alphabets = [chr(char) for char in range(ord('A'), ord('Z') + 1) if char not in [ord('O')]]
    labels = test_dataset.unique_labels
    base_features = {}
    for char in labels:
        cur_char,label = test_dataset.get_random_batch(1,[char])
        cur_feature=sess.run(output,feed_dict={input:cur_char})
        base_features[test_dataset.int2str[label[0]]]=cur_feature[0]
    return base_features

def get_dist(feature1,feature2):
    return np.mean((feature1 - feature2) ** 2)

def eval(test_dataset):

    base_features = get_random_base_feature(test_dataset)

    correct_all = 0
    all = 0

    while True:
        cur_char, label=test_dataset.get_random_batch(1)
        cur_feature = sess.run(output, feed_dict={input: cur_char})

        min_dist=np.Inf
        pred_label=None
        for base_label, base_feature in base_features.items():
            dist=get_dist(cur_feature[0],base_feature)
            if dist<min_dist:
                pred_label=base_label
                min_dist=dist
            # print("{}:{}".format(base_label,dist))
        groudtruth=test_dataset.int2str[label[0]]
        is_true = pred_label == groudtruth
        correct_all = correct_all + 1 if is_true else correct_all
        all += 1
        print("{}-->{},accuary:{}".format(groudtruth,pred_label,correct_all / all))
        print("-------------------------------------------")

        # if not is_true:
        #     cv2.imshow("cur_char",cur_char[0])
        #     cv2.waitKey(0)



        #三元组结果显示
        '''correct_all=0
        all=0
        while True:
            anchor_path, similar_path, dissimilar_path = \
                test_dataset.get_next_triplet_paths()

            anchor = test_dataset.process_data(anchor_path)[np.newaxis,:]
            pos = test_dataset.process_data(similar_path)[np.newaxis,:]
            neg = test_dataset.process_data(dissimilar_path)[np.newaxis,:]

            start_time=time.time()
            anchor=sess.run(output,feed_dict={input:anchor})
            print("time:{}".format(time.time() - start_time))

            pos_output = sess.run(output, feed_dict={input: pos})
            neg_output = sess.run(output, feed_dict={input: neg})

            pos_dist=get_dist(anchor,pos_output)
            neg_dist=get_dist(anchor,neg_output)

            print("pos_dist:",pos_dist)
            print("neg_dist:",neg_dist)
            correct_all=correct_all+1 if pos_dist<neg_dist else correct_all
            all+=1
            print("accuary:{}".format(correct_all/all))
            print("-------------------------------------------")

            anchor_show = cv2.imread(anchor_path)
            similar_show = cv2.imread(similar_path)
            dissimilar_show = cv2.imread(dissimilar_path)

            cv2.imshow("anchor", anchor_show)
            cv2.imshow("similar", similar_show)
            cv2.imshow("dissimilar", dissimilar_show)
            cv2.waitKey(0)'''


if __name__=="__main__":
    test_dataset = dataset.dataset(eval_root, batch_size, support_image_extensions,
                                   input_height, input_width, channals)
    eval(test_dataset)
