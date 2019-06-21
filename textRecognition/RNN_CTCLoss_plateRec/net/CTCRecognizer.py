import tensorflow as tf

class CTCRecognizer(object):
    def __init__(self, is_training,num_classes,num_hidden):
        self.is_training=is_training
        self.num_classes=num_classes
        self.num_hidden=num_hidden

    def preprocess(self, inputs):
        #将transpose的操作写在预处理中，这样以后测试时候不用每次都transpose
        tran_inputs=tf.transpose(inputs,[0,2,1,3])

        NORMALIZER=0.017
        processed_inputs = tf.to_float(tran_inputs)
        red, green, blue = tf.split(processed_inputs, num_or_size_splits=3, axis=3)
        preprocessed_input = (tf.multiply(blue, 0.2989) +tf.multiply(green, 0.5870)+
                              tf.multiply(red, 0.1140))* NORMALIZER
        preprocessed_input=tf.squeeze(preprocessed_input,axis=3)
        return preprocessed_input

    def inference(self,inputs,seq_length):

        cell = tf.contrib.rnn.LSTMCell(64, state_is_tuple=True)
        inner_outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_length, time_major=False, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]
        outputs = tf.reshape(inner_outputs, [-1, 64])  # (7680, 64)

        w = tf.Variable(tf.truncated_normal([64, 12], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0., shape=[12]), name="b")
        logits1 = tf.matmul(outputs, w) + b  # (7680, 12)
        # 变换成和标签向量一致的shape
        logits2 = tf.reshape(logits1, [batch_s, -1, 12])  # (64,120, 12)
        logits = tf.transpose(logits2, (1, 0, 2))  # (120, 64, 12) -(max_step,batch_size,num_class)

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

        return loss_mean
