import os

org_image_root="D:/forTensorflow/stackChars/stackChars2Num_all"
dst_txt_label_root=org_image_root+"_txt_labels"

image_types=["train","test"]

for image_type in image_types:
    # org_image_dir = os.path.join(org_image_root,image_type)
    org_image_dir = org_image_root+"_"+image_type
    dst_txt_label_dir = os.path.join(dst_txt_label_root, image_type)

    if not os.path.exists(dst_txt_label_dir):
        os.makedirs(dst_txt_label_dir)

    for file_name in os.listdir(org_image_dir):
        label=file_name.split("_")[0]
        image_file_path=os.path.join(org_image_dir,file_name)
        text_save_path=os.path.join(dst_txt_label_dir,file_name).replace("jpg","txt")
        print(image_file_path,"--",label)
        with open(text_save_path,"w") as fr:
            fr.write(image_file_path+" "+label)




