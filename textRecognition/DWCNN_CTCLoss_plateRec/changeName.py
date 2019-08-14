import os
import shutil

org_files_dir= "D:/forCaffe/textAreaDet/train_org"

save_dir=org_files_dir.replace("org","rename")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for file_name in os.listdir(org_files_dir):
    org_path=os.path.join(org_files_dir,file_name)

    label=file_name.split("_")[0]
    if "O" in label:
        print(org_path)
    suffix=file_name.replace(label,"")

    label_rename = label.replace("O", "0")
    save_path=os.path.join(save_dir,label_rename+suffix)
    shutil.copyfile(org_path,save_path)
    print(save_path)


