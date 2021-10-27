import sys
import pickle # For serializing data
import os.path # For checking whether a file exist
from nltk.stem import PorterStemmer as ps # For stemming and word tokenization

class PositionalIndex:
    def __init__(self):
        pass

    # Removes most special characters and caps
    def preprocess(self,data):
        for p in "!.,:@#$%^&?<>*()[}{]-=;/\"\\\t\n":
            if p in '\n;?:!.,.':
                data = data.replace(p, ' ')
            else:
                data = data.replace(p, '')
        return data.lower()

    # For each file, opens and adds it to the hashmap
    def createPositionalIndex(self,data_list):
        index = {}
        for i in range(len(data_list)):
            key = data_list[i][0]
            text = data_list[i][1]
            # with open(files[i],encoding='utf-8') as f:
            #    doc = [a for a in preprocess(f.read()).split(' ') if a != ""]
            doc = [a for a in self.preprocess(text).split(' ') if a != ""]
            for idx, word in enumerate(doc):
                stemmed = ps().stem(word)
                if not stemmed in index:
                    index[stemmed] = [(key, idx)]
                else:
                    index[stemmed].append((key, idx))
        return index

    # shows a preview based on the positions and the how
    # much text to show around the data found
    def showPreview(self,data_list, positions, radius):
        for i, (doc_id, word_index) in enumerate(positions):
            for data in data_list:
                if data[0] == doc_id:
                    wordArr = [a for a in self.preprocess(data[1]).split(' ') if a != ""]
                    result = " ".join(wordArr[word_index - radius:word_index + radius])
                    print(str(i + 1) + ": ..." + result + "... " + data[0])
                    # print(data[0],":",data[1])
        print()

    def create(self,plain_text_index_file,positional_index_file):
        list_id_text = pickle.load(open(plain_text_index_file, 'rb'))
        print("Processing and serializing data for future use...")
        pi = self.createPositionalIndex(list_id_text)
        with open(positional_index_file, "wb") as f:
            pickle.dump(pi, f)

    def create_with_data(self,list_id_text,positional_index_file):

        print("Processing and serializing data for future use...")
        pi = self.createPositionalIndex(list_id_text)
        with open(positional_index_file, "wb") as f:
            pickle.dump(pi, f)

    def positional_indexing_search_2words(self,positional_index_file,word1, word2):
        pi=pickle.load(open(positional_index_file,"rb"))
        word1 = ps().stem(self.preprocess(word1).replace(' ', ''))
        word2 = ps().stem(self.preprocess(word2).replace(' ', ''))
        matches = []
        for doc1, index1 in pi[word1]:
            for doc2, index2 in pi[word2]:
                if doc1 != doc2: continue
                if index1 == (index2 - 1): matches.append((doc1, index1))
        list_ids=[]
        for i, (doc_id, word_index) in enumerate(matches):
            list_ids.append(doc_id)
        return list_ids

        # showPreview(list_id_text, matches, 5)

    def positional_indexing_search_2words_num(self,positional_index_file,word1, word2, num):
        pi = pickle.load(open(positional_index_file, "rb"))
        matches = []
        word1 = ps().stem(self.preprocess(word1).replace(' ', ''))
        word2 = ps().stem(self.preprocess(word2).replace(' ', ''))
        print("Searching... \n")
        rad = int(num)
        for doc1, index1 in pi[word1]:
            for doc2, index2 in pi[word2]:
                if doc1 != doc2: continue
                abs_pos = abs(index1 - index2)
                # when abs_pos is 0, the word is itself
                if abs_pos <= rad and abs_pos != 0: matches.append((doc1, index1))
        # showPreview(matches, 5 if rad <= 5 else rad)
        return matches

    def test(self,index_file):
        # test search example
        import time
        queries = ['gram-negative bacteria', 'Fungal infection', 'purulent exudate']

        start_time = time.time()
        for q in queries:
            ws = q.split(" ")
            self.positional_indexing_search_2words(index_file,ws[0], ws[1])

        end_time = time.time()

        print("time cost: ", round((end_time - start_time) * 1000, 2), "ms")


