import tensorflow as tf
import tensorflow.contrib.slim as slim

class CTCRecognizer(object):
    def __init__(self, is_training,keep_prop,num_classes,num_hidden):
        self.is_training=is_training
        self.num_classes=num_classes
        self.num_hidden=num_hidden
        self.keep_prop=keep_prop

    def preprocess(self, inputs):
        #将transpose的操作写在预处理中,这样以后测试时候不用每次都transpose
        tran_inputs=tf.transpose(inputs,[0,2,1,3])
        #
        # NORMALIZER=0.017
        # processed_inputs = tf.to_float(tran_inputs)
        # red, green, blue = tf.split(processed_inputs, num_or_size_splits=3, axis=3)
        # preprocessed_input = (tf.multiply(blue, 0.2989) +tf.multiply(green, 0.5870)+
        #                       tf.multiply(red, 0.1140))* NORMALIZER
        #preprocessed_input=tf.squeeze(preprocessed_input,axis=3)
        return tran_inputs

    # def get_variable_with_l2_loss(self, shape, stddev, wl, name):
    #     biases_num=shape[-1]
    #     var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev), name=name+"weights")
    #     biases=tf.get_variable(name+"_biases",shape=[biases_num],dtype=tf.float32)
    #     if wl is not None:
    #         var_l2_loss = tf.multiply(tf.nn.l2_loss(var), wl, name=name+"_l2_loss")
    #         tf.add_to_collection("Loss", var_l2_loss)
    #     return var,biases

    def block(self,inputs,output_channels,stride=1):
        res_channels=output_channels-output_channels/2

        block_net=slim.separable_convolution2d(inputs,None,[3,3],stride=stride)
        block_net=slim.batch_norm(block_net)

        block_net=slim.convolution2d(block_net,output_channels/2,[3,3])

        block_net = slim.batch_norm(block_net)
        res_net=slim.convolution2d(inputs,res_channels,[3,3],stride=stride)
        return tf.concat([block_net,res_net],axis=3)


    def inference(self,inputs,seq_length):
        with slim.arg_scope(
                [slim.convolution2d],
                weights_initializer=slim.initializers.xavier_initializer(),
                biases_initializer=slim.init_ops.zeros_initializer(),
                weights_regularizer=slim.l2_regularizer(0.005),
                activation_fn=None) as sc1:
            with slim.arg_scope(
                    [slim.batch_norm], is_training=self.is_training,
                    activation_fn=tf.nn.relu, fused=True, decay=0.95) as sc2:
                #input:(b,120,40,3)

                conv_net=slim.convolution2d(inputs,16,kernel_size=[3,3],stride=1)
                conv_net = self.block(conv_net, 32, stride=2)  # (b,60,20,32)

                conv_net2=conv_net     # (b,60,20,32)
                conv_net2=slim.avg_pool2d(conv_net2,[3, 3],[2, 2],padding="SAME")
                conv_net2 = slim.dropout(conv_net2, keep_prob=self.keep_prop)
                avg2 = tf.reduce_mean(tf.square(conv_net2))
                conv_net2 = tf.div(conv_net2, avg2)


                conv_net = slim.convolution2d(conv_net, 32, kernel_size=[3, 3], stride=1)
                conv_net1 = self.block(conv_net, 64, stride=1)  # (b,30,10,64)
                conv_net1 = slim.avg_pool2d(conv_net1,[3, 3],[2, 2],padding="SAME")
                conv_net1 = slim.dropout(conv_net1, keep_prob=self.keep_prop)
                avg1 = tf.reduce_mean(tf.square(conv_net1))
                conv_net1 = tf.div(conv_net1, avg1)


                conv_net3 = slim.convolution2d(inputs, 16,kernel_size=[3,3],stride=1)
                conv_net3=self.block(conv_net3,32,stride=4)   #(b,30,10,32)
                conv_net3 = slim.dropout(conv_net3, keep_prob=self.keep_prop)
                avg3 = tf.reduce_mean(tf.square(conv_net3))
                conv_net3 = tf.div(conv_net3, avg3)

                logits =tf.concat([conv_net1,conv_net2,conv_net3],axis=3)
                logits =slim.convolution2d(logits,self.num_classes,kernel_size=[1,1])
                logits = tf.reduce_mean(logits, axis=2)

                # 必须转成(max_time_step,batch_size,num_class)的形式供后面计算CTC Loss
                logits = tf.transpose(logits, (1, 0, 2))
        return logits

    def beam_searcn(self,logits,seq_len,is_merge=False):
        decoded_logits,log_prob=tf.nn.ctc_beam_search_decoder(logits,seq_len,merge_repeated=is_merge)
        return decoded_logits

    def decode_a_seq(self,indexes, spars_tensor,chars):
        decoded = []
        for m in indexes:
            # print("m:",m)
            str_id = spars_tensor[1][m]
            print(m,"---",str_id)
            str = chars[str_id]
            decoded.append(str)
        return decoded #sparse_tensor[0]是N*2的indices

    def decode_sparse_tensor(self,sparse_tensor,chars):
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        # print(sparse_tensor)
        for offset, i_and_index in enumerate(sparse_tensor[0]):  # sparse_tensor[0]是N*2的indices
            i = i_and_index[0]                                   # 一行是一个样本
            # print("i_and_index:",i_and_index)
            if i != current_i:                                   # current_is是当前样本的id
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()                             # current_seq是当前样本预测值在sparse_tensor的values中对应的下标
            current_seq.append(offset)                           # 之后通过下标就可以从sparse_tensor中找到对应的值
        decoded_indexes.append(current_seq)
        result = []
        for index in decoded_indexes:
            result.append(self.decode_a_seq(index, sparse_tensor,chars))
        return result
    def get_edit_distance_mean(self,decoded_logits_placeholder,sparse_labels_placeholder):
        # 计算两个稀疏矩阵代表的序列的编辑距离,在预测和标签在样本数量上长度不匹配时可以作为一种评判模型的指标,没有他可无妨
        edit_distance_mean = tf.reduce_mean(tf.edit_distance(tf.cast(decoded_logits_placeholder[0], tf.int32), sparse_labels_placeholder))
        return edit_distance_mean

    #传入实值,而非Tensor
    def get_accuarcy(self,decoded_logits,sparse_labels,chars):
        #通过稀疏矩阵解析得到最终的预测结果
        sparse_labels_list = self.decode_sparse_tensor(sparse_labels, chars)
        decoded_list=self.decode_sparse_tensor(decoded_logits,chars)
        true_numer = 0

        if len(decoded_list) != len(sparse_labels_list):
            # print("len(decoded_list)", len(decoded_list), "len(sparse_labels_list)", len(sparse_labels_list),
            #       " test and detect length desn't match")
            return None       #edit_distance起作用

        for idx, pred_number in enumerate(decoded_list):
            groundTruth_number = sparse_labels_list[idx]
            cur_correct = (pred_number == groundTruth_number)
            info_str="{}:{}-({}) <-------> {}-({})".\
                format(cur_correct,groundTruth_number,len(groundTruth_number),pred_number,len(pred_number))
            print(info_str)
            if cur_correct:
                true_numer = true_numer + 1

        accuary=true_numer * 1.0 / len(decoded_list)
        return accuary

    # logits:(24, 50, 67)
    # sparse_groundtrouth:是tf.SparseTensor类型,三元组,其中包括(indices, values, shape)
    def loss(self,logits,sparse_groundtrouth,seq_len):
        loss_all=tf.nn.ctc_loss(labels=sparse_groundtrouth,inputs=logits,sequence_length=seq_len)
        loss_mean=tf.reduce_mean(loss_all)

        # tf.add_to_collection("Loss", loss_mean)
        # loss_all = tf.add_n(tf.get_collection("Loss"), name="total_loss")

        #计算正则损失
        # regularization_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss_all = loss_mean+regularization_loss
        return loss_mean
