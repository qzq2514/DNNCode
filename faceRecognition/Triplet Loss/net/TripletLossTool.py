import tensorflow.contrib.slim as slim
import tensorflow as tf

class TripletLossTool():
    def __init__(self):
        pass

    def batch_triplet_loss(self,ahchor_flatten,similar_flatten,dissimilar_flatten,margin):
        #使用三元损失
        pos_dist = tf.reduce_sum(tf.square(ahchor_flatten - similar_flatten), 1)
        neg_dist = tf.reduce_sum(tf.square(ahchor_flatten - dissimilar_flatten), 1)
        margin_pos_neg=pos_dist - neg_dist + margin
        loss = tf.reduce_mean(tf.maximum(margin_pos_neg, 0))
        return loss,pos_dist,neg_dist

    #计算embeddings中两两距离
    def _pairwise_distances(self,embeddings, squared=False):
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        #对角矩阵,获得dot_product对角部分,
        #square_norm(i,i)表示embeddings[i]*embeddings[i]T
        square_norm = tf.diag_part(dot_product)

        # 计算两两embedding的距离:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        #防止出现负数
        distances = tf.maximum(distances, 0.0)

        if not squared:
            #当distances为0的时候,开平方的梯度为无穷大,防止出现这种情况,则加一个小正数
            mask = tf.to_float(tf.equal(distances, 0.0))
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)
            distances = distances * (1.0 - mask)
        return distances

    def _get_anchor_positive_triplet_mask(self,labels):
        # 对角为全1,其余全为0的矩阵,大小为[batch_size,batch_size]
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        # 对角为全0,其余全为1的矩阵,大小为[batch_size,batch_size],保证后面找到的最大距离的同类不是同一个
        indices_not_equal = tf.logical_not(indices_equal)

        #利用广播的机制,判定两两是不是属于同一类别
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        #mask(i,j)=1表示同一类别(但不是同个样本),mask也是[batch_size,batch_size]
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask

    def _get_anchor_negative_triplet_mask(self,labels):

        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        #mask(i,j)=1表示不同类别,(不同类别肯定不是同一个)
        mask = tf.logical_not(labels_equal)

        return mask

    def batch_triplet_hard_loss(self,embeddings,labels, margin, squared=False):

        #embedding两两距离,[batch_size,batch_size]大小
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        # 得到同类别的mask
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = tf.to_float(mask_anchor_positive)
        # 保留同类别距离,不同类别的的距离被设置为0
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

        # 跨列找到距离每个embedding最大的同类样本,得到该最大距离,hardest_positive_dist大小为[batch_size, 1]
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

        #同理:跨列找到距离每个embedding最小的异类样本,得到该最大距离,hardest_positive_dist大小为[batch_size, 1]
        #但是这里对第i行,将和第i个样本同类的样本距离同时加上该行的一个最大值,保证后面找最小的异类样本的时候不会找到该值
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = tf.to_float(mask_anchor_negative)
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

        #计算与每个样本同类且最大的距离和与每个样本异类但最小的距离的差值,即OHEM下的三元损失
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss,hardest_positive_dist,hardest_negative_dist