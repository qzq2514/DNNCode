import os
import cv2
import numpy as np
import sys

org_files_dir=sys.argv[1]           #原数据集所在的文件夹
multiple_num=int(sys.argv[2])       #需要增强的倍数

argument_save_dir=os.path.join(org_files_dir,"Argument")
org_files=os.listdir(org_files_dir)
if not os.path.exists(argument_save_dir):
    os.makedirs(argument_save_dir)

argument_type_array=[0,0]

#center-(w,h)
def rotate(image,angle,center=None,scale=1.0):
    (h,w)=image.shape[0:2]

    if center==None:
        center=(w/2,h/2)
    M=cv2.getRotationMatrix2D(center,angle,scale)
    rotated_img=cv2.warpAffine(image,M,(w,h))
    return rotated_img


def noising(image,per):
    (h,w)=image.shape[:2]
    noise_img=image.copy()    #必须拷贝图像,不然会修改原图像
    noise_num=int(h*w*per)
    for num in range(noise_num):
        x=np.random.randint(0,w)
        y=np.random.randint(0,h)
        if image.ndim ==3:
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            noise_img[y,x]=np.array([r,g,b])
        elif image.ndim ==1:
            v=np.random.randint(0,255)
            noise_img[y,x] = v
    return noise_img

angles=[]
for org_file_name in org_files:
    org_file_path=os.path.join(org_files_dir,org_file_name)
    print(org_file_path)
    if os.path.isdir(org_file_path):
        continue
    file_type=os.path.split(org_file_path)[1].split(".")[1]
    if file_type not in ["jpg","jpeg","png"]:
        continue
    org_img=cv2.imread(org_file_path)

    for num in range(multiple_num):
        argument_id=np.random.randint(0,2)
        if argument_id==0:
            angle=np.random.randint(-10,10)
            while angle in angles:
                angle=np.random.randint(-8,8)
            rotated_img=rotate(org_img,angle)
            save_path=os.path.join(argument_save_dir,org_file_name+
                                   "_rotate_"+str(argument_type_array[0])+"_"+str(angle)+".jpg")
            cv2.imwrite(save_path,rotated_img)
            argument_type_array[0]+=1
            print(save_path)
            # cv2.imshow("rotated_img", rotated_img)

        elif argument_id==1:
            percetage = np.random.randint(0,15)/100
            noise_img=noising(org_img,percetage)

            save_path = os.path.join(argument_save_dir, org_file_name +
                                     "_noise_" + str(argument_type_array[1]) + ".jpg")
            cv2.imwrite(save_path, noise_img)
            argument_type_array[1] += 1
            print(save_path)
            # cv2.imshow("noise_img", noise_img)
        # cv2.imshow("Img", org_img)
        # cv2.waitKey(0)


