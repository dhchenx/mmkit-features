from mmkfeatures.text.text_features_wrapper import  TextFeaturesWrapper

if __name__=="__main__":
    text_features_wrapper = TextFeaturesWrapper()

    corpus = [
        "This is a sample sentence",
        "I am interested in politics",
        "You are a very good software engineer, engineer.",
    ]

    sentence = "It is a sample sentence"

    # vs=text_features_wrapper.get_text_features_using_tfidf(corpus,[sentence,sentence])
    # vs = text_features_wrapper.get_text_features_using_wordembedding(corpus, sentence)
    vs=text_features_wrapper.get_text_features_using_GloVe(sentence,"../data/glove.6B/glove.6B.50d.txt","../data/glove.6B/glove_6B.50d.wordvec.txt")

    # vs = text_features_wrapper.get_text_features_using_wordvector(corpus, sentence)
    print(vs)

