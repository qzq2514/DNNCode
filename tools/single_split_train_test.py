import sys
import os
import random
import shutil

#仅仅分割一个类的文件
org_file_dir=sys.argv[1]   #该类图片所在的文件夹
train_rate=float(sys.argv[2])

print(train_rate)
work_dir=os.path.split(org_file_dir)[0]

test_save_dir=os.path.join(work_dir,"test")
train_save_dir=os.path.join(work_dir,"train")

if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
if not os.path.exists(train_save_dir):
    os.makedirs(train_save_dir)

files=os.listdir(org_file_dir)

total_num=len(files)
train_num=int(total_num*train_rate)
test_num = total_num - train_num
train_file_list = random.sample(files,train_num)
test_file_list = list(set(files).difference(set(train_file_list)))
# print(total_num,len(train_file_list),len(test_file_list))

for train_file in train_file_list:
    train_file_path=os.path.join(train_save_dir,train_file)
    org_file_path=os.path.join(org_file_dir,train_file)
    fileExt=os.path.splitext(org_file_path)[1][1:]
    if fileExt in ["jpg","jpeg","png"]:
        print(train_file_path)
        shutil.copyfile(org_file_path,train_file_path)


for test_file in test_file_list:
    test_file_path=os.path.join(test_save_dir,test_file)
    org_file_path=os.path.join(org_file_dir,test_file)
    fileExt = os.path.splitext(org_file_path)[1][1:]
    if fileExt in ["jpg", "jpeg", "png"]:
        print(test_file_path)
        shutil.copyfile(org_file_path,test_file_path)
