import tensorflow as tf
from tensorflow.python.framework import graph_util
from net import siameseNet_batchnorm as siameseNet
import dataset
import numpy as np
import cv2
import os

batch_size=64
input_height=32
input_width=32
total_epoch_num=1000
snapshot=100
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]
margin=2
channals=3
# "P:/WorkSpace/VS2015/FaceRecognition/FaceRecognition/trainData/faceData/dataBase41"
train_image_root="P:/WorkSpace/VS2015/FaceRecognition/FaceRecognition/trainData/faceData/dataBase41"
test_image_root="P:/WorkSpace/VS2015/FaceRecognition/FaceRecognition/trainData/faceData/dataBase41"

model_path="models/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")

if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="siamese_face_bn"

if __name__ == '__main__':
    print("model_name:",model_name)

    input_left_placeholder = tf.placeholder(tf.float32,shape=[None,input_height,input_width,channals],name="input_left")
    input_right_placeholder = tf.placeholder(tf.float32, shape=[None, input_height, input_width, channals], name="input_right")
    pair_label_placeholder = tf.placeholder(tf.int32, shape=[None], name="pair_label")

    is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
    pair_label_placeholder = tf.to_float(pair_label_placeholder)

    siamese_net=siameseNet.siameseNet()
    left_out = siamese_net.inference(input_left_placeholder,reuse=False,is_training = is_training)
    right_out = siamese_net.inference(input_right_placeholder,reuse=True,is_training = is_training)
    loss , dist= siamese_net.loss(left_out,right_out,pair_label_placeholder,margin)

    flatten_out_left = tf.identity(left_out, name="flatten_out")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.9)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    with tf.control_dependencies([tf.group(*update_ops)]):
        # train_step = optimizer.minimize(loss, global_step)
        train_step = tf.train.MomentumOptimizer(0.01, 0.90). \
            minimize(loss, global_step=global_step)

    var_list = tf.trainable_variables()
    if global_step is not None:
        var_list.append(global_step)
    g_list = tf.global_variables()  # 从全局变量中获得batch norm的缩放和偏差
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    ckpt_saver = tf.train.Saver()
    train_dataset = dataset.dataset(train_image_root,batch_size,support_image_extensions,
                    input_height,input_width,channals)

    test_dataset = dataset.dataset(test_image_root, batch_size, support_image_extensions,
                                    input_height, input_width, channals)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_iters_num = 0
        for epoch_num in range(total_epoch_num):

            train_images_num = train_dataset.sample_len
            cur_epoch_iters_num = train_images_num // batch_size
            for iters_num in range(cur_epoch_iters_num):

                train_images_left, train_images_right, train_pair_label = \
                    train_dataset.next_pair_batch()

                test_images_left, test_images_right, test_pair_label = \
                    test_dataset.next_pair_batch()

                if train_images_left is None or test_images_left is None:
                    continue
                train_dict = {input_left_placeholder: train_images_left,
                              input_right_placeholder: train_images_right,
                              pair_label_placeholder: train_pair_label,
                              is_training: True}
                test_dict =  {input_left_placeholder: test_images_left,
                              input_right_placeholder: test_images_right,
                              pair_label_placeholder: test_pair_label,
                              is_training: False}
                sess.run(train_step, feed_dict=train_dict)

                _train_loss, _train_dist = \
                    sess.run([loss, dist], feed_dict=train_dict)
                _test_loss, _test_pos_dist = \
                    sess.run([loss, dist], feed_dict=test_dict)

                print("distance:", list(zip(train_pair_label, _train_dist.flatten()))[:5])
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = np.mean(one_moving_meaning.eval())
                print("one_moving_meaning:", one_moving_meaning_show)

                total_iters_num += 1
                if total_iters_num % snapshot == 0:
                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["flatten_out"])
                    with tf.gfile.FastGFile(pb_path + model_name + "-" + str(total_iters_num) + ".pb", mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    ckpt_saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=total_iters_num)
                show_text = "epoch:{},epoch-iters:{},total-iters:{},loss:{},lr:{},val:{}".format \
                    (epoch_num, iters_num + 1, total_iters_num, _train_loss, "0.99",_test_loss)
                print(show_text)



