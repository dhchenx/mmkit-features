
class MMFeaturesNode(dict):
    '''
    Multimodal Features Node Structure

    Content Id:
        name: a short name for the content
        format: '', a string indicating which formatio of the content
        text: '', a long text to describe the content
        modality: '', a string indicate which modality of the content
        feature_ids: [], ids for each features, if [] then using feature index in link representation
        feature_dim_names: [[]], if empty then using using index as dim name
        feature_extractor: a registered name to extract features from the original file
        features: an numpy array [[]]
        intervals: [[]] starting time and ending time, where [] indicates not specific
        space: [length, width,height], default value [] represents not specific
        labels: a list of labels, useful for multi-label or binary classification;a single label represent all feature has the same label
        origin: "" the original content file path, optional
        attributes: dict{}, user-defined key-value properties
        links: a list of relationships between the content's features, between inside and outside content
            e.g. link format: (feat_1,(rel_direction,rel_type,rel_func),feat_2),
                    where feat_1,feat_2 can be feature ids outside current content using format: content:feat1
    '''
    def __init__(self, content_id,meta_fields=None):
        super().__init__()
        self.data={}
        self["content_id"]=content_id
        if meta_fields!=None and type(meta_fields)==dict:
            for k in meta_fields.keys():
                self[k]=meta_fields[k]
            klist = list(self.keys())
            for k in klist:
                if type(self[k]) == str and (self[k] == "" or self[k] == None):
                    self.pop(k)
                elif type(self[k]) == list and len(self[k]) == 0:
                    self.pop(k)
                elif type(self[k]) == dict and len(self[k].keys()) == 0:
                    self.pop(k)
            # print(self)


    def get_content_id(self):
        return self["content_id"]

    def get_all_validated_keys(self):
        return ["content_id",'name','text','modality','feature_ids','feature_dim_names',
                'features','feature_extractor','intervals','space','labels','label_names',
                'origin','attributes','links',"format","raw","locations","objects"
                ]

    def get_all_validated_keys_alias(self):
        return ["ID", 'NA', 'TX', 'MD', 'FID', 'FDM',
                'F', 'FET', 'ITV', 'SP', 'LB', 'LBN',
                'OG', 'AT', 'LK', "FM","RAW","LOC","OBJ"
                ]

    def set_item(self,k,v):
        if k in self.get_all_validated_keys() or k in self.get_all_validated_keys_alias():
            self[k]=v
        else:
            raise Exception(f"{k} is not in validated key list!")

    def map_to_short_keys(self):
        temp_dict={}
        for k in self.keys():
            temp_dict[k]=self[k]
        for k in temp_dict.keys():
            index=self.get_all_validated_keys().index(k)
            self[self.get_all_validated_keys_alias()[index]]=temp_dict[k]
            self.pop(k)

    def validate_empty_field(self):
        klist=list(self.keys())
        for k in klist:
            if type(self[k]) == str and (self[k] == "" or self[k]==None) :
                self.pop(k)
            elif type(self[k]) == list and len(self[k]) == 0:
                self.pop(k)
            elif type(self[k]) == dict and len(self[k].keys()) == 0:
                self.pop(k)


    def get_item(self,k):
        return self[k]



