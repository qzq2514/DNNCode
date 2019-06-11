import cv2
import os
from numpy import *
from scipy import * 
import numpy as np 


image_dir = "D:/plateData/projectPlate/orgData/WI_QI/All/usefulImgs/WI"
label_dir = "D:/plateData/projectPlate/orgData/WI_QI/All/Labels/WI"

save_img_dir="D:/forTensorflow/plateLandmarkDetTrain2/images"
save_label_dir="D:/forTensorflow/plateLandmarkDetTrain2/labels"

label_ext="json"

warp_times=4

def main():
	for label_file in os.listdir(label_dir):
		if not label_file.endswith(label_ext):
			continue
		label_path=os.path.join(label_dir, label_file)
		image_file = label_file.replace(label_ext, "jpg")
		img_path=os.path.join(image_dir, image_file)

		if not os.path.exists(img_path):
			continue

		org_image = cv2.imread(img_path)
		org_img_heigth, org_img_width = org_image.shape[:2]

		#根据label的存储格式进行解析信息-只要保证最后infos中存储的是四个角点八个整型坐标数据就行
		#四个点的顺序是左上-右上-右下-左下

		with open(label_path) as f:
			infos = eval(f.read())["region"]

		corners = []
		org_xmin = np.inf
		org_ymin = np.inf
		org_xmax = -np.inf
		org_ymax = -np.inf
		for corner in infos:
			x = corner["x"]
			y = corner["y"]

			org_xmin = min(org_xmin, x)
			org_ymin = min(org_ymin, y)
			org_xmax = max(org_xmax, x)
			org_ymax = max(org_ymax, y)
			corners.append([x, y])

		org_plate_width = org_xmax - org_xmin
		org_plate_height = org_ymax - org_ymin

		corners=np.array(corners)

		corners_random_list=[]
		for warp_id in range(warp_times):

			save_img_name=str(warp_id)+image_file
			save_lab_name=str(warp_id)+label_file

			cur_corners_random=[]
			img_show=org_image.copy()

			random_corner_Xmax = -np.inf
			random_corner_Ymax = -np.inf
			random_corner_Xmin = np.inf
			random_corner_Ymin = np.inf

			for corner in corners:
				x=corner[0]
				y=corner[1]

				cv2.circle(img_show, (x, y), 2, (0, 0, 255), 2)
				random_x = x + org_plate_width * np.random.randint (-10,10)/100
				random_y = y + org_plate_height * np.random.randint (-10,10)/100

				#随机角点信息
				random_x = int(min(max(0, random_x),org_img_width))
				random_y = int(min(max(0, random_y), org_img_heigth))

				#随机角点后的最小外包矩形信息
				random_corner_Xmax = int(max(random_corner_Xmax, random_x))
				random_corner_Ymax = int(max(random_corner_Ymax, random_y))
				random_corner_Xmin = int(min(random_corner_Xmin, random_x))
				random_corner_Ymin = int(min(random_corner_Ymin, random_y))

				cur_corners_random.append([random_x,random_y])

			cur_corners_random=np.array(cur_corners_random,dtype=np.float32)

			corners = np.float32(corners)
			M = cv2.getPerspectiveTransform(corners, cur_corners_random)
			dst_img = cv2.warpPerspective(org_image, M, (org_img_width, org_img_heigth))

			def_img_show=dst_img.copy()

			for random_corner in cur_corners_random:
				x=random_corner[0]
				y=random_corner[1]
				cv2.circle(def_img_show,(x,y),2,(0,255,0),3)

			# 随机角点最小外包矩形的扩充
			random_corner_Xmax_loose = int(random_corner_Xmax + np.random.randint(10, 50) / 100 * org_plate_width)
			random_corner_Ymax_loose = int(random_corner_Ymax + np.random.randint(10, 80) / 100 * org_plate_height)
			random_corner_Xmin_loose = int(random_corner_Xmin - np.random.randint(10, 50) / 100 * org_plate_width)
			random_corner_Ymin_loose = int(random_corner_Ymin - np.random.randint(10, 80) / 100 * org_plate_height)

			random_corner_Xmax_loose = min(random_corner_Xmax_loose, org_img_width)
			random_corner_Ymax_loose = min(random_corner_Ymax_loose, org_img_heigth)
			random_corner_Xmin_loose = max(random_corner_Xmin_loose, 0)
			random_corner_Ymin_loose = max(random_corner_Ymin_loose, 0)

			plate_img_loose = dst_img[random_corner_Ymin_loose:random_corner_Ymax_loose,
							  random_corner_Xmin_loose:random_corner_Xmax_loose]

			plate_img_loose_show=plate_img_loose.copy()

			cur_corners_random_relate = []
			for random_corner in cur_corners_random:
				x = random_corner[0]
				y = random_corner[1]

				relate_x = int(x - random_corner_Xmin_loose)
				relate_y = int(y - random_corner_Ymin_loose)
				cv2.circle(plate_img_loose_show,(relate_x,relate_y),2,(255,0,0),3)
				cur_corners_random_relate.extend([relate_x,relate_y])

			save_str=str(cur_corners_random_relate)
			print(save_str)

			#保存信息
			with open(os.path.join(save_label_dir,save_lab_name),"w") as fw:
				fw.write(save_str)
			cv2.imwrite(os.path.join(save_img_dir,save_img_name),plate_img_loose)

			# cv2.imshow("img_show",img_show)
			# cv2.imshow("def_img_show", def_img_show)
			# cv2.imshow("plate_img_loose", plate_img_loose)
			# cv2.imshow("plate_img_loose_show", plate_img_loose_show)
			# cv2.waitKey(0)

if __name__ == '__main__':
	main()