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
total_epoch_num=50
snapshot=100
support_image_extensions=[".jpg",".png",".jpeg",".bmp"]
margin=1.0
channals=3

train_image_root="D:/forTensorflow/charRecTrain/forMyDNNCode/train"
test_image_root="D:/forTensorflow/charRecTrain/forMyDNNCode/test"

model_path="models/"
pb_path=os.path.join(model_path,"pb/")
ckpt_path=os.path.join(model_path,"ckpt/")

if not os.path.exists(pb_path):
    os.makedirs(pb_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
model_name="siamese_triplet_28out_allloss_bn"

if __name__ == '__main__':
    # image_paths,labels=get_images_path(test_image_root)
    # data=next_batch(True,None,image_paths,labels)
    # for left,right,label in zip(*data):
    #     cv2.imshow("left",left)
    #     cv2.imshow("right", right)
    #     print(label)
    #     cv2.waitKey(0)

    first_shape=None
    anchor_placeholder = tf.placeholder(tf.float32,shape=[first_shape,input_height,input_width,channals],name="anchor")
    similar_placeholder = tf.placeholder(tf.float32, shape=[first_shape, input_height, input_width, channals], name="similar")
    dissimilar_placeholder = tf.placeholder(tf.float32, shape=[first_shape, input_height, input_width, channals], name="dissimilar")
    labels_placeholder = tf.placeholder(tf.float32, shape=
                                             [None if first_shape is None else first_shape * 3, ], name="labels")
    is_training_placeholder = tf.placeholder_with_default(False, shape=(), name="is_training")
    siamese_net=siameseNet.siameseNet()

    anchor = siamese_net.inference(anchor_placeholder,reuse=False,is_training=is_training_placeholder)
    similar = siamese_net.inference(similar_placeholder,reuse=True,is_training=is_training_placeholder)
    dissimilar = siamese_net.inference(dissimilar_placeholder,reuse=True,is_training=is_training_placeholder)
    loss,pos_dist,neg_dist = siamese_net.loss(anchor,similar,dissimilar,labels_placeholder,margin)

    flatten_out_anchor = tf.identity(anchor, name="flatten_anchor")
    flatten_out_similar = tf.identity(similar, name="flatten_similar")
    flatten_out_dissimilar = tf.identity(dissimilar, name="flatten_dissimilar")

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.9)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    with tf.control_dependencies([tf.group(*update_ops)]):
        # train_step = optimizer.minimize(loss, global_step)
        train_step = tf.train.MomentumOptimizer(0.01, 0.90).\
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

        # if os.path.exists(os.path.join(ckpt_path, "checkpoint")):
        #     ckpt_saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        total_iters_num = 0
        for epoch_num in range(total_epoch_num):

            train_images_num = train_dataset.sample_len
            cur_epoch_iters_num = train_images_num // batch_size
            for iters_num in range(cur_epoch_iters_num):

                train_anchor, train_similar, train_dissimilar,train_labels = \
                    train_dataset.next_triplet_batch()
                test_anchor, test_similar, test_dissimilar,test_labels = \
                    test_dataset.next_triplet_batch()

                if train_anchor is None or test_anchor is None:
                    continue
                train_dict = {anchor_placeholder: train_anchor,
                              similar_placeholder: train_similar,
                              dissimilar_placeholder: train_dissimilar,
							  labels_placeholder:train_labels,
                              is_training_placeholder:True}
                test_dict = {anchor_placeholder: test_anchor,
                             similar_placeholder: test_similar,
                             dissimilar_placeholder: test_dissimilar,
							 labels_placeholder:test_labels,
                             is_training_placeholder: False}
                _,_global_step=sess.run([train_step,global_step], feed_dict=train_dict)

                anchor_out,similar_out,dissimilar_out = sess.run([
                    flatten_out_anchor,flatten_out_similar,flatten_out_dissimilar],
                    feed_dict=train_dict)

                _train_loss,_train_pos_dist,_train_neg_dist = \
                    sess.run([loss,pos_dist,neg_dist], feed_dict=train_dict)
                _test_loss,_test_pos_dist,_test_neg_dist =\
                    sess.run([loss,pos_dist,neg_dist], feed_dict=test_dict)

                print("distance:",list(zip(_train_pos_dist.flatten(),_train_neg_dist.flatten()))[:5])
                one_moving_meaning_show = "No mean or variance"
                if len(bn_moving_vars) > 0:
                    one_moving_meaning = sess.graph.get_tensor_by_name(bn_moving_vars[0].name)
                    one_moving_meaning_show = "{}={}".\
                        format(bn_moving_vars[0].name,np.mean(one_moving_meaning.eval()))

                print(one_moving_meaning_show)
                show_text = "epoch:{},epoch-iters:{},total-iters:{},loss:{},lr:{},val:{}".format \
                    (epoch_num, iters_num + 1, _global_step, _train_loss, "0.99", _test_loss)
                print(show_text)

                if _global_step % snapshot == 0:
                    # 保存PB
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["flatten_anchor"])
                    save_model_name=model_name + "-" + str(_global_step) + ".pb"
                    with tf.gfile.FastGFile(pb_path + save_model_name, mode="wb") as fw:
                        fw.write(constant_graph.SerializeToString())
                    # 保存CKPT
                    ckpt_saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=total_iters_num)
                    print("Successfully saved model {}".format(save_model_name))




