import numpy as np
from mmkfeatures.fusion.computational_sequence_x import computational_sequence_x
from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
import h5py
import nexusformat.nexus as nx
# import ast
from tqdm import tqdm
import pickle
import cv2
from quickcsv import *

class MMFeaturesLib:
    def __init__(self,root_name="",dataset_name="",file_path="",description="",version="1.0",creator="mmkit-features",contact="",enable_key_compressed=False):
        if file_path=="" or file_path==None:
            self.root_name = root_name
            self.name = dataset_name
            self.meta_dict = {
                "root name": self.root_name,  # name of the featureset
                "computational sequence description": description,  # name of the featureset
                "dimension names": None,
                "computational sequence version": "1.0",  # the version of featureset
                "alignment compatible": "",  # name of the featureset
                "dataset name": self.name,  # featureset belongs to which dataset
                "dataset version": version,  # the version of dataset
                "creator": creator,  # the author of the featureset
                "contact": contact,  # the contact of the featureset
                "featureset bib citation": "",  # citation for the paper related to this featureset
                "dataset bib citation": "",  # citation for the dataset,
                "attributes_alias":"",
                "attributes":""
            }
            self.compseq = computational_sequence_x(self.name)
            self.compseq_data = {}
            self.compseq_rel_data={}
            self.compseq_dicts={}
            self.feature_dimension_num = 1
        else:
            self.compseq = computational_sequence_x(file_path)
            self.meta_dict=self.compseq.metadata
            self.root_name=self.meta_dict["root name"]
            self.name=self.meta_dict["dataset name"]
            self.compseq_data=self.compseq.data
            self.compseq_rel_data=self.compseq.rel_data
            self.compseq_dicts={}
            self.file_path=file_path
        self.use_alias_names=enable_key_compressed
        if self.use_alias_names:
            self.meta_dict["compressed"]="True"
            temp = MMFeaturesNode("")
            self.meta_dict["attributes"] = ','.join(temp.get_all_validated_keys())
            self.meta_dict["attributes_alias"] = ','.join(temp.get_all_validated_keys_alias())
        else:
            self.meta_dict["compressed"]="False"
            self.meta_dict["attributes"] = ""
            self.meta_dict["attributes_alias"] = ""

    def use_alias_name(self):
        return self.use_alias_names

    def set_compressed(self,enable=True):
        temp=MMFeaturesNode("")
        self.meta_dict["attributes"]=','.join(temp.get_all_validated_keys())
        self.meta_dict["attributes_alias"]=','.join(temp.get_all_validated_keys_alias())
        self.meta_dict["compressed"]=str(enable)
        self.use_alias_names=enable

    def set_features_name(self,feature_names):
        self.feature_names=feature_names
        self.feature_dimension_num=len(feature_names)
        self.meta_dict["dimension names"]=feature_names

    '''
    def set_data(self,features):
        self.compseq_data = {}
        ids=features.keys()
        for key in ids:
            self.compseq_data[key] = {}
            self.compseq_data[key]["features"] = np.array(features[key][0])
            # let's assume each video is one minute, hence 60 seconds.
            self.compseq_data[key]["intervals"] = np.array(features[key][1])
        self.compseq.setData(self.compseq_data, self.name)
    '''

    def compress_content(self,content):
        new_content=MMFeaturesNode(content.get_content_id())
        ks=content.keys()
        ks_ori=content.get_all_validated_keys()
        mapped_ks=content.get_all_validated_keys_alias()
        for k in ks:
            new_k=mapped_ks[ks_ori.index(k)]
            new_content.set_item(new_k,content[k])
        new_content.pop("content_id")
        return new_content

    def set_rel_data(self,rel_data):
        self.compseq_rel_data=rel_data
        self.compseq.setRelData(self.compseq_rel_data)

    def set_dicts(self,dicts):
        self.compseq_dicts=dicts
        self.compseq.setDicts(self.compseq_dicts)

    def set_data(self,list_content):

        if self.use_alias_names:
            self.compseq_data = {}
            ks=list(list_content.keys())
            for k in ks:
                self.compseq_data[k]=self.compress_content(list_content[k])
        else:
            self.compseq_data = list_content

        # self.compseq_data = list_content
        self.compseq.setData(self.compseq_data, self.name)

    def get_data(self):
        return self.compseq

    def save_data(self,f_path=None):
        if f_path==None:
            f_path=self.name+".csd"
        self.compseq.deploy(f_path, my_metadata=self.meta_dict)

    def show_structure(self,f_path=None):
        if f_path == None:
            file_name = self.file_path
        else:
            file_name = f_path
        f = nx.nxload(file_name)
        print(f.tree)
        print()

    def show_sample_data_compressed(self,group,max_num):
        meta_keys = list(group["metadata"])
        metadata = group["metadata"]
        print("=====================Start Meta Data======================")
        for mk in meta_keys:
            print(f"{mk} -> {group['metadata'][mk][0]}")
        print("=====================End Meta Data====================")
        print("=====================Start Content Data====================")
        content_ids = list(group["data"])
        for k in content_ids[:max_num]:  # each video
            print("ID: ", k)
            print(list(group["data"][k]))
            content_keys=list(group["data"][k])
            part = group["data"][k]
            if 'F' in content_keys:
                features = part['F']
            else:
                features=[]
            if 'ITV' in content_keys:
                intervals = part['ITV']
            else:
                intervals=[]
            print("[features]: ")
            for f in features[:5]:
                print("\t", f)
            print("number of content features: ", len(features))
            print("[intervals]: ")
            for inter in intervals[:5]:
                print("\t", inter)
            print("number of content intervals: ", len(intervals))
            for kk in part.keys():
                if kk not in ["F", "ITV"]:
                    # print(kk)
                    print(kk, ":  ", part[kk][()])
                    if kk in ["LB", "FID", "LBN"]:
                        labels = part[kk]
                        # print("type of labels",type(labels.value))
                        # print("type of label list", type(labels_list))
                        # print(part[kk].value)
                        for item in eval(part[kk][()]):
                            print("\t\t--> ", item)
                    if kk in ['AT']:
                        for k in eval(part[kk][()]).keys():
                            print("\t\t-->  ", k, " = ", eval(part[kk][()])[k])
        print()
        print("Printing...relationship table")
        rel_ids = list(group["rel_data"])
        for rel_id in rel_ids[:max_num]:
            rel_obj = group["rel_data"][rel_id]
            print(rel_obj["start"][()], "-->", rel_obj["end"][()])

    def show_sample_data_not_compressed(self,group,max_num):
        meta_keys = list(group["metadata"])
        metadata = group["metadata"]
        print("=====================Start Meta Data======================")
        for mk in meta_keys:
            print(f"{mk} -> {group['metadata'][mk][0]}")
        print("=====================End Meta Data====================")
        if "dicts" in group.keys():
            dict_data=group["dicts"]
            for d in dict_data:
                print(d)
                dd=dict_data[d]
                count=0
                for d1 in dd:
                    count+=1
                    print("  "+str(d1)+"  --> ",dd[d1][()])
                    if count>=5:
                        break
                print("  ...")


        print("=====================Start Content Data====================")
        content_ids = list(group["data"])
        for k in content_ids[:max_num]:  # each video
            print("[Content ID]: ", k)
            part = group["data"][k]
            keys=list(group["data"][k])
            if 'features' in keys:
                features = part['features']
            else:
                features=[]
            if 'intervals' in keys:
                intervals = part['intervals']
            else:
                intervals=[]
            print("[features]: ")
            for f in features[:5]:
                print("\t", f)
            print("number of content features: ", len(features))
            print("[intervals]: ")
            for inter in intervals[:5]:
                print("\t", inter)
            print("number of content intervals: ", len(intervals))
            for kk in part.keys():
                if kk not in ["features", "intervals"]:
                    # print(kk)
                    print(kk, ":  ", part[kk][()])
                    if kk in ["labels", "feature_ids", "label_names"]:
                        labels = part[kk]
                        # print("type of labels",type(labels.value))
                        # print("type of label list", type(labels_list))
                        # print(part[kk].value)
                        for item in eval(part[kk][()]):
                            print("\t\t--> ", item)
                    if kk in ['attributes']:
                        print(part[kk][()])
                        for k in eval(part[kk][()]).keys():
                            print("k = ",k)
                            print("\t\t-->  ", k, " = ", eval(part[kk][()])[k])
        print()
        print("Printing...relationship table")
        rel_ids = list(group["rel_data"])
        for rel_id in rel_ids[:max_num]:
            rel_obj=group["rel_data"][rel_id]
            print(rel_obj["start"][()], "-->", rel_obj["end"][()])

    def show_sample_data(self,f_path=None,max_num=5):
        if f_path==None:
            file_name=self.file_path
        else:
            file_name=f_path
        print()
        print("MMF FileName: ",file_name)
        with h5py.File(file_name, "r") as f:
            group = f[self.root_name]
            if 'compressed' in list(group["metadata"]):
                compressed=eval(str(group["metadata"]["compressed"][0]))
            else:
                compressed="false"
            # str(metadata_[key][0])
            is_compressed=False
            if compressed == "True" or compressed == "true":
                is_compressed=True

            if not is_compressed:
                self.show_sample_data_not_compressed(group,max_num)
            else:
                self.show_sample_data_compressed(group,max_num)

    def to_index_file(self,field,index_file_path,index_type="brutal_force"):

        data=self.get_data()
        if index_type=="brutal_force":
            list_id_value = []
            for key in tqdm(data.keys()):
                content = data[key]
                if field in content.keys():
                    v = content[field][()]
                else:
                    v=""
                # print(key,v)
                list_id_value.append([key, v])
            pickle.dump(list_id_value, open(index_file_path, 'wb'))
        if index_type=="inverted_index":
            from mmkfeatures.index.index_inverted_op import InvertedIndex
            inverted_index = InvertedIndex()
            list_id_value = []
            for key in tqdm(data.keys()):
                content = data[key]
                if field in content.keys():
                    v = content[field][()]
                else:
                    v = ""
                list_id_value.append([key, v])
            inverted_index.create_with_data(list_id_value,index_file_path)
        if index_type=="positional_index":
            from mmkfeatures.index.index_positional_op import PositionalIndex
            pos_index = PositionalIndex()
            list_id_value = []
            for key in tqdm(data.keys()):
                content = data[key]
                if field in content.keys():
                    v = content[field][()]
                else:
                    v = ""
                list_id_value.append([key, v])
            pos_index.create_with_data(list_id_value,index_file_path)

    def search_index(self,index_file_path,query,search_type="brutal_force"):
        list_ids = []
        if search_type=="brutal_force":
            list_id_text = pickle.load(open(index_file_path, 'rb'))
            for item in list_id_text:
                text = str(item[1])
                if query in text:
                    list_ids.append(item[0])
            return list_ids
        if search_type=="inverted_index":
            from mmkfeatures.index.index_inverted_op import InvertedIndex
            inverted_index=InvertedIndex()
            return list(inverted_index.search(index_file_path,query))
        if search_type=="positional_index":
            from mmkfeatures.index.index_positional_op import PositionalIndex
            pos_index=PositionalIndex()
            ws=query.split(" ")
            return pos_index.positional_indexing_search_2words(index_file_path,ws[0],ws[1])
        return None

    def to_obj_index(self,index_file,obj_field="objects",index_type="color_descriptor"):
        print("Creating index....")
        if index_type=="color_descriptor":
            from mmkfeatures.image.color_descriptor import ColorDescriptor
            cd = ColorDescriptor((8, 12, 3))
            data = self.get_data()
            f_out = open(index_file, "w")
            for cid in tqdm(data.keys()):
                item = data[cid]
                imgs = item[obj_field]
                for img_id in imgs:
                    # print(image)
                    img = imgs[img_id][()]
                    # print(img)
                    # print(type(img))
                    features = cd.describe(img)
                    features = [str(f) for f in features]
                    # print(feature)
                    feature_str = cid + "," + ",".join(features)
                    f_out.write(feature_str + "\n")
            f_out.close()
        elif index_type=="autoencoder":
            pass

    def export_key_values(self,field,save_path=""):
        data=self.get_data()
        list_item=[]
        for key in tqdm(data.keys()):
            content = data[key]
            if field in content.keys():
                v = content[field][()]
            else:
                v = ""
            if type(v)!=str:
                v=v.decode("utf-8",errors="ignore")
            list_item.append({"key":key,"value":v})
        if save_path!="":
            write_csv(save_path=save_path,list_rows=list_item)
        return list_item


    def search_obj_index(self,index_file,features):
        from mmkfeatures.image.color_descriptor import ColorDescriptor
        from mmkfeatures.image.image_searcher import Searcher
        # initialize the image descriptor
        print("Searching index....")
        # perform the search
        searcher = Searcher(index_file)
        results = searcher.search(features)
        # display the query
        # cv2.imshow("Query", query)
        # cv2.waitKey(0)
        return results

    def get_content_by_id(self,cid):
        return self.get_data()[cid]

