import sys
import os
import random
import shutil


#仅仅分割一个类的文件
org_file_root="D:/forTensorflow/plateLandmarkDetTrain2/CA" #sys.argv[1]   #该类图片和标签所在的根文件夹
train_rate=0.8 #float(sys.argv[2])


def split(org_file_root,train_rate):

    test_save_root=os.path.join(org_file_root,"test")
    train_save_root=os.path.join(org_file_root,"train")

    dir_names=[]
    for dir_name in os.listdir(org_file_root):
        if dir_name!="train" and dir_name!="test":
            dir_names.append(dir_name)

    if len(dir_names)==0:
        print("There is no dir in this root!")
        return

    base_dir=os.path.join(org_file_root,dir_names[0])
    base_file_names=os.listdir(base_dir)   #基准文件名(不带后缀)

    if len(base_file_names)==0:
        print("There is no files in these dir!")
        return

    base_file_names_no=[os.path.splitext(base_file_name)[0] for base_file_name in base_file_names]

    total_num = len(base_file_names_no)
    train_num=int(total_num*train_rate)
    test_num = total_num - train_num

    train_file_names_no = random.sample(base_file_names_no,train_num)   #训练集基准文件名(不带后缀)
    test_file_names_no = list(set(base_file_names_no).difference(set(train_file_names_no)))   #测试集基准文件名(不带后缀)

    #开始分
    for dir_name in dir_names:

        cur_save_train_dir = os.path.join(train_save_root,dir_name)
        cur_save_test_dir = os.path.join(test_save_root, dir_name)
        if not os.path.exists(cur_save_train_dir):
            os.makedirs(cur_save_train_dir)
        if not os.path.exists(cur_save_test_dir):
            os.makedirs(cur_save_test_dir)


        cur_dir=os.path.join(org_file_root,dir_name)
        files=os.listdir(cur_dir)
        cur_ext=os.path.splitext(files[0])[1]

        for train_file_name_no in train_file_names_no:
            cur_train_file_name = train_file_name_no+cur_ext
            cur_train_file_path = os.path.join(cur_dir,cur_train_file_name)

            save_train_file_path = os.path.join(cur_save_train_dir,cur_train_file_name)
            shutil.copyfile(cur_train_file_path,save_train_file_path)
            print("{}/train-->{}".format(dir_name,save_train_file_path))

        for test_file_name_no in test_file_names_no:
            cur_test_file_name = test_file_name_no+cur_ext
            cur_test_file_path = os.path.join(cur_dir,cur_test_file_name)

            save_test_file_path = os.path.join(cur_save_test_dir,cur_test_file_name)
            shutil.copyfile(cur_test_file_path,save_test_file_path)
            print("{}/test-->{}".format(dir_name,save_test_file_path))

if __name__ == '__main__':

    split(org_file_root,train_rate)








# if not os.path.exists(test_save_dir):
#     os.makedirs(test_save_dir)
#     os.makedirs(os.path.join(test_save_dir,"images"))
#     os.makedirs(os.path.join(test_save_dir, "labels"))
#
# if not os.path.exists(train_save_dir):
#     os.makedirs(train_save_dir)
#     os.makedirs(os.path.join(train_save_dir, "images"))
#     os.makedirs(os.path.join(train_save_dir, "labels"))
#
# org_images_dir=os.path.join(org_file_root,"images")
# org_labels_dir=os.path.join(org_file_root,"labels")
#
# image_files=os.listdir(org_images_dir)
#
# total_num=len(image_files)
# train_num=int(total_num*train_rate)
# test_num = total_num - train_num
# train_file_list = random.sample(image_files,train_num)
# test_file_list = list(set(image_files).difference(set(train_file_list)))
# # print(total_num,len(train_file_list),len(test_file_list))
#
#
# for train_image_name in train_file_list:
#     train_image_path=os.path.join(train_save_dir,"images",train_image_name)
#     org_image_path=os.path.join(org_images_dir,train_image_name)
#
#     train_label_path=os.path.join(org_labels_dir,"test",train_image_name)
#
#     fileExt=os.path.splitext(org_file_path)[1][1:]
#     if fileExt in ["jpg","jpeg","png"]:
#         print(train_file_path)
#         shutil.copyfile(org_file_path,train_file_path)
#
#
# for test_file in test_file_list:
#     test_file_path=os.path.join(test_save_dir,test_file)
#     org_file_path=os.path.join(org_file_dir,test_file)
#     fileExt = os.path.splitext(org_file_path)[1][1:]
#     if fileExt in ["jpg", "jpeg", "png"]:
#         print(test_file_path)
#         shutil.copyfile(org_file_path,test_file_path)
