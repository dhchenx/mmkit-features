import os
from shutil import copyfile
import pickle
from tqdm import tqdm
# load images id
dict_image_ids={}
root_path=r"D:\数据集\医学数据集\roco-datasets\all_data"

# load semtypes
list_class_tag=[]
list_class_names=[]

dict_image_semtypes={}

def load_semtypes(path):
    f_train_semtypes = open(path, encoding='utf-8')
    line = f_train_semtypes.readline()
    while line:
        line = line.replace('\r', '').strip()
        if line == '':
            line = f_train_semtypes.readline()
            continue
        fs = line.split('\t')
        if len(fs) < 2:
            line = f_train_semtypes.readline()
            continue
        roco_id = fs[0]
        id=int(roco_id.split("_")[1])
        ls = fs[2:]
        if len(ls) < 2:
            line = f_train_semtypes.readline()
            continue
        # print(roco_id, ls)
        list_sem=[]
        for x in range(0,len(ls)-2,2):
            class_tag = ls[x]
            class_name = ls[x + 1]
            if class_tag not in list_class_tag:
                list_class_tag.append(class_tag)
                list_class_names.append(class_name)
            list_sem.append(class_tag)
        dict_image_semtypes[roco_id]=list_sem
        print(roco_id,list_sem)
        line = f_train_semtypes.readline()

    f_train_semtypes.close()

train_semtypes_path=root_path+"/train/radiology/semtypes.txt"
test_semtypes_path=root_path+"/test/radiology/semtypes.txt"
val_semtypes_path=root_path+"/validation/radiology/semtypes.txt"

print("loading semtypes...")
load_semtypes(train_semtypes_path)
load_semtypes(test_semtypes_path)
load_semtypes(val_semtypes_path)

pickle.dump(dict_image_semtypes,open("datasets/dict_image_semtypes.pickle","wb"))

