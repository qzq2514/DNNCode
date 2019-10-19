import numpy as np
import cv2
import os


class dataset():
    def __init__(self,image_root,batch_size,support_image_extensions,
                 input_height,input_width,channals):
        self.image_root = image_root
        self.batch_size = batch_size
        self.support_image_extensions=support_image_extensions
        self.input_height = input_height
        self.input_width = input_width
        self.channals=channals
        self.channal_flag = cv2.IMREAD_GRAYSCALE if self.channals == 1 else cv2.IMREAD_COLOR
        # image_paths:[None,input_height,input_width,channals]
        # labels:[None,]
        self.image_paths,self.labels=self.get_images_path()
        self.sample_len=len(self.image_paths)
        self.unique_labels=np.unique(self.labels)
        self.class_dict = {v: k for k, v in enumerate(self.unique_labels)}
        self.int2str = {k: v for k, v in enumerate(self.unique_labels)}

    def get_similar_pair(self):
        index = np.random.choice(self.sample_len, 1)[0]
        similar_path = np.random.choice(self.image_paths[self.labels
                                                         == self.labels[index]], 1)[0]
        # print(labels[index])
        return self.image_paths[index], similar_path

    def get_dissimilar_pair(self):
        index = np.random.choice(self.sample_len, 1)[0]
        similar_path = np.random.choice(self.image_paths[self.labels
                                                         !=self. labels[index]], 1)[0]
        # print(labels[index])
        return self.image_paths[index], similar_path

    def process_data(self,image_path):
        image = cv2.imread(image_path, self.channal_flag)
        if image is None:
            return None

        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_np_resized = np.resize(image_resized,
                                          (self.input_height, self.input_width, self.channals))
        return image_np_resized

    def get_one_pair_path(self):
        pair_label = 1
        if (np.random.rand() > 0.5):   #相似样本,标签为1
            left_path, right_path = self.get_similar_pair()
        else:                          #不相似样本,标签为0
            left_path, right_path = self.get_dissimilar_pair()
            pair_label = 0
        return left_path,right_path,pair_label

    def get_images_path(self):
        image_paths = []
        labels = []

        for label_str in os.listdir(self.image_root):
            class_dir = os.path.join(self.image_root, label_str)
            for file_name in os.listdir(class_dir):
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in self.support_image_extensions:
                    continue
                image_path = os.path.join(self.image_root, label_str, file_name)
                image_paths.append(image_path)
                labels.append(label_str)

        data_path_array = np.array([image_paths, labels])  # 2xN大小,其中N是总样本数
        data_path_array = data_path_array.transpose()

        np.random.shuffle(data_path_array)
        image_paths = list(data_path_array[:, 0])
        labels = data_path_array[:, 1]

        # image_paths和labels都是列向量形式
        return np.array(image_paths), np.array(labels)

    def get_all_data(self):
        images_data=[]
        for image_path in self.image_paths:
            image = self.process_data(image_path)
            images_data.append(image)
        return np.array(images_data)

    def get_random_batch(self,random_batch_size,target_labels=None):
        if target_labels==None:
            target_labels=self.unique_labels
        index=[label in target_labels for label in self.labels]
        index=np.array(index)
        cur_samples_path=self.image_paths[index]
        cur_labels = self.labels[index]

        random_batch_data = []
        index_batch=np.random.choice(len(cur_samples_path),random_batch_size)
        for image_path in cur_samples_path[index_batch]:
            image = self.process_data(image_path)
            random_batch_data.append(image)
        res_int_labels = [self.class_dict[label] for label in cur_labels[index_batch]]
        return np.array(random_batch_data),res_int_labels

    def next_pair_batch(self):
        batch_image_left = []
        batch_image_right = []
        batch_pair_label = []

        for _ in range(self.batch_size):
            left_path, right_path,pair_label=self.get_one_pair_path()

            image_left = self.process_data(left_path)
            image_right = self.process_data(right_path)
            if image_left is None or image_right is None:
                continue

            batch_image_left.append(image_left)
            batch_image_right.append(image_right)
            batch_pair_label.append(pair_label)
        batch_image_left=np.array(batch_image_left)
        batch_image_right=np.array(batch_image_right)        #获取原始数据,归一化等在模型中做
        # batch_pair_label=np.expand_dims(np.array(batch_pair_label),axis=1)
        # batch_pair_label=np.array(batch_pair_label)[:,np.newaxis]
        return batch_image_left, batch_image_right,batch_pair_label

if __name__ == '__main__':
    test_image_root = "D:/forTensorflow/charRecTrain/forMyDNNCode/test"
    batch_size = 64
    input_height = 28
    input_width = 28
    support_image_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
    channals = 3

    test_dataset = dataset(test_image_root, batch_size, support_image_extensions,
                                      input_height, input_width, channals)
    # infos = test_dataset.get_random_batch(12,['B','C','D'])
    # for image,label in zip(*infos):
    #     print(image)
    #     cv2.imshow("image",image)
    #     cv2.waitKey(0)

    while True:
        left_path,right_path,label=test_dataset.get_one_pair_path()
        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)
        print(label)
        cv2.imshow("left_image",left_image)
        cv2.imshow("right_image", right_image)
        cv2.waitKey(0)
