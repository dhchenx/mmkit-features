import pickle
import csv

root_path=r"D:\数据集\医学数据集\roco-datasets\all_data"

# load semtypes
list_class_tag=[]
list_class_names=[]

# dict_image_class={}
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
        # print(roco_id,list_sem)
        line = f_train_semtypes.readline()

    f_train_semtypes.close()

list_image_file={}
list_image_text={}

def load_image_file_path(path):
    print(path)
    with open(path,encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            # print(row)
            roco_id=row[0]
            if roco_id=='id':
                continue
            # id=int(roco_id.split("_")[1])
            file_name=row[1]
            text=row[2].replace("\n","")
            text=text.strip()
            # print('id:',id)
            # print("---------------")
            list_image_file[roco_id]=file_name
            list_image_text[roco_id]=text

list_image_keywords={}

def load_image_keywords_path(path):

    print(path)
    with open(path,encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            # print(row)
            roco_id=row[0]
            if roco_id=='id':
                continue
            keywords=row[1:]
            # print(roco_id,keywords)
            list_image_keywords[roco_id]=keywords

list_image_lic={}

def load_image_lic_path(path):
    print(path)
    with open(path,encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for row in f_csv:

            print(row)
            roco_id=row[0]
            if roco_id=='id':
                continue
            pmc_id=row[1]
            cc=row[2]
            # print(roco_id,keywords)
            list_image_lic[roco_id]=cc
            list_image_file[roco_id]=pmc_id

list_image_cuis={}
def load_image_cuis_path(path):
    print(path)
    with open(path,encoding='utf-8')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            # print(row)
            roco_id=row[0]
            if roco_id=='id':
                continue
            cuis=row[1:]

            # print(roco_id,keywords)
            list_image_cuis[roco_id]=cuis

list_image_caption={}
def load_image_caption_path(path):
    print(path)
    lines=open(path,encoding='utf-8').readlines()
    for line in lines:
        line=line.strip()
        ls=line.split("\t")

        # print(ls)
        roco_id=ls[0].strip()
        if len(ls)==2:
            caption=ls[1].strip()
        else:
            caption=""
        list_image_caption[roco_id]=caption
        list_image_text[roco_id]=caption

list_image_link={}
def load_image_link_path(path):
    print(path)
    lines=open(path,encoding='utf-8').readlines()
    for line in lines:
        line=line.strip()
        ls=line.split("\t")

        # print(ls)
        roco_id=ls[0].strip()
        link=ls[1].strip()
        name=ls[2].strip()
        list_image_link[roco_id]=[link,name]

train_semtypes_path=root_path+"/train/non-radiology/semtypes.txt"
test_semtypes_path=root_path+"/test/non-radiology/semtypes.txt"
val_semtypes_path=root_path+"/validation/non-radiology/semtypes.txt"

print("loading semtypes...")
load_semtypes(train_semtypes_path)
load_semtypes(test_semtypes_path)
load_semtypes(val_semtypes_path)

# data
'''
print()
path_train=root_path+"/train/non-radiology/traindata.csv"
path_test=root_path+"/test/non-radiology/testdata.csv"
path_val=root_path+"/validation/non-radiology/valdata.csv"
load_image_file_path(path_train)
load_image_file_path(path_test)
load_image_file_path(path_val)
'''

# keywords
print()
path_k_train=root_path+"/train/non-radiology/keywords.txt"
path_k_test=root_path+"/test/non-radiology/keywords.txt"
path_k_val=root_path+"/validation/non-radiology/keywords.txt"
load_image_keywords_path(path_k_train)
load_image_keywords_path(path_k_test)
load_image_keywords_path(path_k_val)

# license
print()
path_lic_train=root_path+"/train/non-radiology/licences.txt"
path_lic_test=root_path+"/test/non-radiology/licences.txt"
path_lic_val=root_path+"/validation/non-radiology/licences.txt"
load_image_lic_path(path_lic_train)
load_image_lic_path(path_lic_test)
load_image_lic_path(path_lic_val)

# cuis
print()
path_cuis_train=root_path+"/train/non-radiology/cuis.txt"
path_cuis_test=root_path+"/test/non-radiology/cuis.txt"
path_cuis_val=root_path+"/validation/non-radiology/cuis.txt"
load_image_cuis_path(path_cuis_train)
load_image_cuis_path(path_cuis_test)
load_image_cuis_path(path_cuis_val)

# captions
print()
path_caption_train=root_path+"/train/non-radiology/captions.txt"
path_caption_test=root_path+"/test/non-radiology/captions.txt"
path_caption_val=root_path+"/validation/non-radiology/captions.txt"
load_image_caption_path(path_caption_train)
load_image_caption_path(path_caption_test)
load_image_caption_path(path_caption_val)

# links
print()
path_d_train=root_path+"/train/non-radiology/dlinks.txt"
path_d_test=root_path+"/test/non-radiology/dlinks.txt"
path_d_val=root_path+"/validation/non-radiology/dlinks.txt"
load_image_link_path(path_d_train)
load_image_link_path(path_d_test)
load_image_link_path(path_d_val)

pickle.dump(dict_image_semtypes,open("datasets/non-radiology/list_image_semtypes.pickle","wb"))
pickle.dump(list_class_tag,open("datasets/non-radiology/list_class_tag.pickle","wb"))
pickle.dump(list_class_names,open("datasets/non-radiology/list_class_name.pickle","wb"))
pickle.dump(list_image_file,open("datasets/non-radiology/list_image_file.pickle","wb"))
pickle.dump(list_image_text,open("datasets/non-radiology/list_image_text.pickle","wb"))
pickle.dump(list_image_keywords,open("datasets/non-radiology/list_image_keywords.pickle","wb"))
pickle.dump(list_image_lic,open("datasets/non-radiology/list_image_lic.pickle","wb"))
pickle.dump(list_image_cuis,open("datasets/non-radiology/list_image_cuis.pickle","wb"))
pickle.dump(list_image_caption,open("datasets/non-radiology/list_image_caption.pickle","wb"))
pickle.dump(list_image_link,open("datasets/non-radiology/list_image_link.pickle","wb"))