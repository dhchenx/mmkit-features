from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from mmkfeatures.text.featurization import Bow
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

class TextFeaturesWrapper:

    def __init__(self):
        pass

    def get_text_features_using_tfidf(self,corpus, text):
        # Create CountVectorizer, which create bag-of-words model.
        # stop_words : Specify language to remove stopwords.
        vectorizer = TfidfVectorizer(stop_words='english')

        # Learn vocabulary in sentences.
        vectorizer.fit(corpus)

        # Get dictionary.
        vectorizer.get_feature_names()

        # Transform each sentences in vector space.
        if type(text) == str:
            input_text = [text]
        else:
            input_text = text
        vector = vectorizer.transform(input_text)
        vector_spaces = vector.toarray()
        if type(text) == str:
            return vector_spaces[0]
        else:
            return vector_spaces

    def get_text_features_using_bow(self,corpus, text):
        # Create CountVectorizer, which create bag-of-words model.
        # stop_words : Specify language to remove stopwords.
        vectorizer = CountVectorizer(stop_words='english')

        # Learn vocabulary in sentences.
        vectorizer.fit(corpus)

        # Get dictionary.
        vectorizer.get_feature_names()

        # Transform each sentences in vector space.
        if type(text) == str:
            input_text = [text]
        else:
            input_text = text

        vector = vectorizer.transform(input_text)
        vector_spaces = vector.toarray()
        if type(text) == str:
            return vector_spaces[0]
        else:
            return vector_spaces

    def tokenize(self,sentence):
        return [token.lower() for token in sentence.split() if token not in STOP_WORDS]


    def get_text_features_using_bow2(self,corpus, text):
        # sentences = ['this is a list of sentences', 'second sentence in list of sentences', 'a word for complexity']
        bow = Bow()
        bow.fit(corpus)
        bow.transform(corpus)
        vs = bow.transform(text)
        return vs

    def get_text_features_using_wordvector(self,corpus, text):

        new_corpus = []
        for c in corpus:
            new_corpus.append(' '.join(self.tokenize(c)))

        model = Word2Vec(sentences=new_corpus, size=100, window=5, min_count=1, workers=4)
        # model.save("word2vec.model")
        # model = Word2Vec.load("word2vec.model")

        # model.train(new_corpus, total_examples=1, epochs=1)
        # vector = model.wv['computer']  # get numpy vector of a word
        # sims = model.wv.most_similar('computer', topn=10)  # get other similar words
        sen_tokens = self.tokenize(text)
        print("tokens: ", sen_tokens)
        vs = []
        for w in sen_tokens:
            vs.append(model.wv[w])
        return vs

    def get_text_features_using_GloVe(self,text, glove_input_file, word2vec_output_file):
        if not os.path.exists(word2vec_output_file):
            print("Converting glove file to wordvec file... ")
            (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
        # load model
        glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

        sen_tokens = self.tokenize(text)
        print("tokens: ", sen_tokens)
        vs = []
        for w in sen_tokens:
            vs.append(glove_model[w])
        return vs
        # obtain the vector of a single word
        # cat_vec = glove_model['cat']
        # print(cat_vec)
        # get similar words
        # print(glove_model.most_similar('frog'))
        # return cat_vec

if __name__ == "__main__":

    text_features_wrapper=TextFeaturesWrapper()

    corpus = [
        "This is a sample sentence",
        "I am interested in politics",
        "You are a very good software engineer, engineer.",
    ]

    sentence= "It is a sample sentence"

    # vs=get_text_features_using_tfidf(corpus,[sentence,sentence])
    # vs = get_text_features_using_wordembedding(corpus, sentence)
    # vs=get_text_features_using_GloVe(sentence,"../../data/glove.6B/glove.6B.50d.txt","../../data/glove.6B/glove_6B.50d.wordvec.txt")

    vs=text_features_wrapper.get_text_features_using_wordvector(corpus,sentence)
    print(vs)
