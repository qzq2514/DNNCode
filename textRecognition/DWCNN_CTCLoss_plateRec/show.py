import os
import cv2
import numpy as np

train_dir = "D:/forCaffe/textAreaDet/train_rename"


for file_name in os.listdir(train_dir):
    label=file_name.split("_")[0]
    print("label:",label)
    file_path=os.path.join(train_dir,file_name)

    img=cv2.imread(file_path)

    # blue, green, red = np.split(img,3,axis=2)
    blue, green, red  = cv2.split(img)

    # print(blue[0,0],green[0,0],red[0,0])
    mergr_img=(blue*0.2989+green*0.5870+red*0.1140)/255

    mergr_img=mergr_img*255
    mergr_img=mergr_img.astype(np.uint8)

    mergr_img[mergr_img>255]=255
    # print(mergr_img)

    if len(label)!=7:
        cv2.imshow("mergr_img",mergr_img)
        cv2.waitKey(0)