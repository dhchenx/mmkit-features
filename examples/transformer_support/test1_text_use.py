from mmkfeatures.transformer.text.model import *

if __name__=="__main__":
    root_folder = "datasets/text1"
    input_text = "lung cancer"
    start = time.time()
    fe=TransformerTextFeatureExtractor(model_or_path="transfo-xl-wt103",
                                       data_path=f'{root_folder}/data1.csv',
                                       feature_data_path=f'{root_folder}/feature_list1.npy',
                                       max_num=100)
    print("creating...")
    fe.create()
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
