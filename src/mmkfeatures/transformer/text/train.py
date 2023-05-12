from transformers import TransfoXLConfig, TransfoXLModel,TransfoXLTokenizer
import pandas as pd
import torch
import numpy as np
import scipy.spatial.distance as distance

# 18-layer, 1024-hidden, 16-heads, 257M parameters.
# English model trained on wikitext-103
# Initializing a Transformer XL configuration

configuration = TransfoXLConfig().from_pretrained("transfo-xl-wt103")
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# Initializing a model from the configuration
model = TransfoXLModel.from_pretrained("transfo-xl-wt103", config = configuration)

## extract the features
dataset = pd.read_csv('data/data.csv')[:10]
print(dataset.shape)
pages = dataset['desp'].values.tolist()
print("the dataset is:\n", pages)

saved_features = []
for val in pages:
    input_ids = torch.tensor(tokenizer.encode(val, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0] # dimension is (1, sequence length, hidden states)
    # average the hidden states to get the semantic content of the input
    extracted_features = torch.mean(last_hidden_states, 1)[0].detach().numpy() # now the dimension is (1, sequencelength, 1)
    saved_features.append(extracted_features)

np.save("data/feature_list.npy",saved_features,allow_pickle=True)

##
features_list = np.load("data/feature_list.npy",allow_pickle=True)

input_text = "Childhood Obesity"
print("the input text is:", input_text)

input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0] # dimension is (1, sequence length, hidden states)
# average the hidden states to get the semantic content of the input
extracted_features = torch.mean(last_hidden_states, 1)[0].detach().numpy()

def cal_sim(input_fe,seed_fe):
    return 1 - distance.cosine(np.array(input_fe), np.array(seed_fe))

sim_socre = []
# cosine similarity
for seed_feature in features_list:
    sim_socre.append(cal_sim(extracted_features,seed_feature))

idx = sim_socre.index(max(sim_socre))
print("similarity scores of all:\n",sim_socre)
val = pages[idx]
print("the highest similar data in the dataset:",val)


