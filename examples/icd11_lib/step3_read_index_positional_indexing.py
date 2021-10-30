import sys
import pickle # For serializing data
import os.path # For checking whether a file exist
from nltk.stem import PorterStemmer as ps # For stemming and word tokenization

list_id_text=pickle.load(open("icd11_mmf/index/index_id_text.pickle",'rb'))

# Removes most special characters and caps
def preprocess(data):
    for p in "!.,:@#$%^&?<>*()[}{]-=;/\"\\\t\n":
        if p in '\n;?:!.,.':
            data = data.replace(p,' ')
        else: data = data.replace(p,'')
    return data.lower()

# For each file, opens and adds it to the hashmap
def createPositionalIndex(data_list):
    index = {}
    for i in range(len(data_list)):
        key=data_list[i][0]
        text=data_list[i][1]
        #with open(files[i],encoding='utf-8') as f:
        #    doc = [a for a in preprocess(f.read()).split(' ') if a != ""]
        doc=[a for a in preprocess(text).split(' ') if a !=""]
        for idx, word in enumerate(doc):
            stemmed = ps().stem(word)
            if not stemmed in index:
                index[stemmed] = [(key,idx)]
            else: index[stemmed].append((key,idx))
    return index

# shows a preview based on the positions and the how
# much text to show around the data found
def showPreview(data_list,positions,radius):
    for i, (doc_id, word_index) in enumerate(positions):
        for data in data_list:
            if data[0]==doc_id:
                wordArr = [a for a in preprocess(data[1]).split(' ') if a != ""]
                result = " ".join(wordArr[word_index - radius:word_index + radius])
                print(str(i + 1) + ": ..." + result + "... " + data[0])
                # print(data[0],":",data[1])
    print()

# Serialization/Positional Index

index_file="icd11_mmf/index/index_id_text_positional.pickle"
# Serialization/Positional Index
pi = {}
if os.path.isfile(index_file):
    print("Loading data...")
    with open(index_file,"rb") as f:
        pi = pickle.load(f)
else:
    print("Processing and serializing data for future use...")
    pi = createPositionalIndex(list_id_text)
    with open(index_file,"wb") as f:
        pickle.dump(pi,f)

def positinal_indexing_search_2words(word1,word2):

    word1 = ps().stem(preprocess(ws[0]).replace(' ', ''))
    word2 = ps().stem(preprocess(ws[1]).replace(' ', ''))
    print("Searching... \n")
    matches = []
    for doc1, index1 in pi[word1]:
        for doc2, index2 in pi[word2]:
            if doc1 != doc2: continue
            if index1 == (index2 - 1): matches.append((doc1, index1))

    # showPreview(list_id_text, matches, 5)

def positinal_indexing_search_2words_num(word1,word2,num):
    matches = []
    word1 = ps().stem(preprocess(word1).replace(' ', ''))
    word2 = ps().stem(preprocess(word2).replace(' ', ''))
    print("Searching... \n")
    rad = int(num)
    for doc1, index1 in pi[word1]:
        for doc2, index2 in pi[word2]:
            if doc1 != doc2: continue
            abs_pos = abs(index1 - index2)
            # when abs_pos is 0, the word is itself
            if abs_pos <= rad and abs_pos != 0: matches.append((doc1, index1))
    # showPreview(matches, 5 if rad <= 5 else rad)

# test search example
import time
queries = ['gram-negative bacteria', 'Fungal infection', 'purulent exudate']

start_time=time.time()
for q in queries:
    ws = q.split(" ")
    positinal_indexing_search_2words(ws[0],ws[1])

end_time=time.time()

print("time cost: ",round((end_time-start_time)*1000,2),"ms")

