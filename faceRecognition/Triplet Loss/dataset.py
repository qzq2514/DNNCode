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
        self.class_dict= {v:k for k,v in enumerate(self.unique_labels)}
        self.int2str = {k: v for k, v in enumerate(self.unique_labels)}

    def get_next_triplet_paths(self):

        index = np.random.choice(self.sample_len, 1)[0]
        similar_path = np.random.choice(self.image_paths[self.labels
                                                         == self.labels[index]], 1)[0]
        dissimilar_mask = self.labels!= self.labels[index]
        dissimilar_indexs = np.array(range(self.sample_len))[dissimilar_mask]
        dissimilar_index = np.random.choice(dissimilar_indexs, 1)[0]

        dissimilar_path = self.image_paths[dissimilar_index]

        labels = np.array([self.labels[index],self.labels[index],self.labels[dissimilar_index]])
        return self.image_paths[index], similar_path,dissimilar_path,labels

    def process_data(self,image_path):
        image = cv2.imread(image_path, self.channal_flag)
        if image is None:
            return None
        image_resized = cv2.resize(image, (self.input_width, self.input_height))
        image_np_resized = np.resize(image_resized,
                                          (self.input_height, self.input_width, self.channals))
        return image_np_resized

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

    #获取数据和对应的真实标签,用于画聚类图
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
        res_int_labels=[self.class_dict[label] for label in cur_labels[index_batch]]
        return np.array(random_batch_data),res_int_labels

    #获取三元组
    def next_triplet_batch(self):
        anchor_images = []
        similar_images = []
        dissimilar_images = []
        batch_labels={"anchor":[],"pos":[],"neg":[]}
        for _ in range(self.batch_size):
            anchor_path, similar_path,dissimilar_path,cue_labels\
                =self.get_next_triplet_paths()

            anchor_image = self.process_data(anchor_path)
            similar_image = self.process_data(similar_path)
            dissimilar_image = self.process_data(dissimilar_path)
            if anchor_image is None or similar_image is None or dissimilar_image is None:
                continue

            anchor_images.append(anchor_image)
            similar_images.append(similar_image)
            dissimilar_images.append(dissimilar_image)
            batch_labels["anchor"].append(self.class_dict[cue_labels[0]])
            batch_labels["pos"].append(self.class_dict[cue_labels[1]])
            batch_labels["neg"].append(self.class_dict[cue_labels[2]])

        anchor_images=np.array(anchor_images)
        similar_images=np.array(similar_images)        #获取原始数据,归一化等在模型中做
        dissimilar_images=np.array(dissimilar_images)
        return anchor_images, similar_images,dissimilar_images,\
               np.array([*batch_labels["anchor"],*batch_labels["pos"],*batch_labels["neg"]])

if __name__ == '__main__':
    test_image_root = "D:/forTensorflow/charRecTrain/forMyDNNCode/test"
    batch_size = 64
    input_height = 28
    input_width = 28
    support_image_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
    channals = 3

    test_dataset = dataset(test_image_root, batch_size, support_image_extensions,
                                      input_height, input_width, channals)

    # while(True):
    #     infos = test_dataset.next_triplet_batch()

    # infos = test_dataset.next_triplet_batch()
    # labels = infos[-1]
    # index=0
    # for anchor,similar_image,dissimilar_image in zip(*(infos[:-1])):
    #     print(labels[index],labels[index+batch_size],labels[index+batch_size*2])
    #     index+=1
    #     cv2.imshow("anchor",anchor)
    #     cv2.imshow("similar_image", similar_image)
    #     cv2.imshow("dissimilar_image", dissimilar_image)
    #
    #     cv2.waitKey(0)

    # while True:
    #     anchor_path, similar_path, dissimilar_path,lables = \
    #         test_dataset.get_next_triplet_paths()
    #
    #     print(lables)
    #     anchor = cv2.imread(anchor_path)
    #     similar = cv2.imread(similar_path)
    #     dissimilar = cv2.imread(dissimilar_path)
    #
    #     cv2.imshow("anchor",anchor)
    #     cv2.imshow("similar", similar)
    #     cv2.imshow("dissimilar", dissimilar)
    #     cv2.waitKey(0)


    image_datas,labels = test_dataset.get_random_batch(12,["A","B","C","D"])
    for image_data,label in list(zip(image_datas,labels)):
        cv2.imshow("image_data", image_data)
        print("label:",label)
        cv2.waitKey(0)
