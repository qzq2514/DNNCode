import sys
import os
import random
import shutil

org_file_root=sys.argv[1]             #原始图片集的文件夹(该文件夹下是每个类的图片的文件夹)
train_rate=float(sys.argv[2])         #训练集比例
print(train_rate)
work_dir=os.path.split(org_file_root)[0]

test_save_root=os.path.join(work_dir,"test")
train_save_root=os.path.join(work_dir,"train")
if not os.path.exists(test_save_root):
    os.makedirs(test_save_root)
if not os.path.exists(train_save_root):
    os.makedirs(train_save_root)

for root,dir,files in os.walk(org_file_root):
    if root==org_file_root:
        continue
    tag=os.path.split(root)[1]
    test_save_dir=os.path.join(test_save_root,tag)
    train_save_dir = os.path.join(train_save_root, tag)

    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)

    total_num=len(files)
    train_num=int(total_num*train_rate)
    test_num = total_num - train_num
    train_file_list = random.sample(files,train_num)
    test_file_list = list(set(files).difference(set(train_file_list)))
    # print(total_num,len(train_file_list),len(test_file_list))

    for train_file in train_file_list:
        train_file_path=os.path.join(train_save_dir,train_file)
        org_file_path=os.path.join(root,train_file)
        shutil.copyfile(org_file_path,train_file_path)
        print(train_file_path)

    for test_file in test_file_list:
        test_file_path=os.path.join(test_save_dir,test_file)
        org_file_path=os.path.join(root,test_file)
        shutil.copyfile(org_file_path,test_file_path)
        print(test_file_path)