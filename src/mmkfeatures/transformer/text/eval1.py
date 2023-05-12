from transformers import TransfoXLConfig, TransfoXLModel,TransfoXLTokenizer
import pandas as pd
import torch
import numpy as np
import scipy.spatial.distance as distance
import time
from tqdm import tqdm
# 18-layer, 1024-hidden, 16-heads, 257M parameters.
# English model trained on wikitext-103
# Initializing a Transformer XL configuration

class TransformerTextFeatureExtractor:

    def __init__(self,data_path):
        self.configuration = TransfoXLConfig().from_pretrained("transfo-xl-wt103")
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        # Initializing a model from the configuration
        self.model = TransfoXLModel.from_pretrained("transfo-xl-wt103", config=self.configuration)
        self.data_path=data_path
        ## extract the features
        dataset = pd.read_csv(self.data_path)[:100]
        print(dataset.shape)
        self.pages = dataset['text'].values.tolist()
        # print("the dataset is:\n", self.pages)

    def create(self):



        saved_features = []
        for val in tqdm(self.pages):
            input_ids = torch.tensor(self.tokenizer.encode(val, add_special_tokens=True)).unsqueeze(0)
            outputs = self.model(input_ids)
            last_hidden_states = outputs[0] # dimension is (1, sequence length, hidden states)
            # average the hidden states to get the semantic content of the input
            extracted_features = torch.mean(last_hidden_states, 1)[0].detach().numpy() # now the dimension is (1, sequencelength, 1)
            saved_features.append(extracted_features)

        np.save(f"{root_folder}/feature_list.npy",saved_features,allow_pickle=True)

    def load(self):

        ##
        self.features_list = np.load(f"{root_folder}/feature_list.npy",allow_pickle=True)

    def cal_sim(self,input_fe,seed_fe):
        return 1 - distance.cosine(np.array(input_fe), np.array(seed_fe))

    def search(self,input_text):

        print("the input text is:", input_text)

        input_ids = torch.tensor(self.tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0] # dimension is (1, sequence length, hidden states)
        # average the hidden states to get the semantic content of the input
        extracted_features = torch.mean(last_hidden_states, 1)[0].detach().numpy()

        sim_socre = []
        # cosine similarity
        for seed_feature in self.features_list:
            sim_socre.append(self.cal_sim(extracted_features,seed_feature))

        idx = sim_socre.index(max(sim_socre))
        print("similarity scores of all:\n",sim_socre)
        val = self.pages[idx]
        print("the highest similar data in the dataset:",val)


if __name__=="__main__":
    root_folder = "data1"
    input_text = "lung cancer"
    start = time.time()
    fe=TransformerTextFeatureExtractor(data_path=f'{root_folder}/{root_folder}.csv')

    print("creating...")
    # fe.create()
    d1=time.time()-start
    start=time.time()
    print("loading...")
    fe.load()
    d2 = time.time() - start
    start = time.time()
    print("searching...")
    fe.search(input_text=input_text)
    d3 = time.time() - start
    start = time.time()
    print("d1=",d1)
    print("d2=",d2)
    print("d3=",d3)
