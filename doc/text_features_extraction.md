## Text Features Extraction

Extracting text features is quite important in natural language processing. Therefore, the ```mmkit-features``` toolkit allows us to easily obtain text features in different format using `TextFeaturesWrapper` class. Several text feature extracting techniques are implemented in the module, including using `TF-IDF`, `WordEmbedding`, `WordVector` and `GloVe`. 

Here are a few toy examples to show its usage. 

```python
from mmkfeatures.text.text_features_wrapper import  TextFeaturesWrapper

if __name__=="__main__":

    text_features_wrapper = TextFeaturesWrapper()

    corpus = [
        "This is a sample sentence",
        "I am interested in politics",
        "You are a very good software engineer, engineer.",
    ]

    sentence = "It is a sample sentence"
    # 1. Using TFIDF
    # vs=text_features_wrapper.get_text_features_using_tfidf(corpus,[sentence,sentence])

    # 2. Using WordEmbedding
    # vs = text_features_wrapper.get_text_features_using_wordembedding(corpus, sentence)
    
    # 3. Using GloVe Embedding
    vs=text_features_wrapper.get_text_features_using_GloVe(sentence,"../data/glove.6B/glove.6B.50d.txt","../data/glove.6B/glove_6B.50d.wordvec.txt")

    # 4. Using Word Vector
    # vs = text_features_wrapper.get_text_features_using_wordvector(corpus, sentence)

    print(vs)

```

Most of the methods generate word vectors with fixed length to represent text for our analysis. We highly recommend you to use GloVe embedding to generate word vectors. 