from mmkfeatures.fusion.computational_sequence import computational_sequence
from mmkfeatures.fusion.dataset import mmdataset
import numpy
import objgraph

def random_init(compseq, feat_dim):
    for vid_key in vid_keys:
        num_entries = numpy.random.randint(low=5, high=100, size=1).astype(int)[0]
        feat_dim1 = feat_dim.astype(int)[0]
        print("num_entries:", num_entries)
        print("fea_dim:", feat_dim)
        compseq[vid_key] = {}
        compseq[vid_key]["features"] = numpy.random.uniform(low=0, high=1, size=[num_entries, feat_dim1])
        # let's assume each video is one minute, hence 60 seconds.
        compseq[vid_key]["intervals"] = numpy.arange(start=0, stop=60 + 0.000001,
                                                     step=60. / ((2 * num_entries) - 1)).reshape([num_entries, 2])


if __name__ == "__main__":
    vid_keys = ["video1", "video2", "video3", "video4", "video5", "Hello", "World", "UG3sfZKtCQI"]

    # let's assume compseq_1 is some modality with a random feature dimension
    compseq_1_data = {}
    compseq_1_feature_dim = numpy.random.randint(low=20, high=100, size=1)
    random_init(compseq_1_data, compseq_1_feature_dim)

    compseq_1 = computational_sequence("my_compseq_1")
    compseq_1.setData(compseq_1_data, "my_compseq_1")

    # objgraph.show_backrefs([compseq_1], filename='sample-graph.png')
    import random
    objgraph.show_chain(
        objgraph.find_backref_chain(
            random.choice(objgraph.by_type('computational_sequence')),
            objgraph.is_proper_module),
        filename='chain.png')

    # let's assume compseq_2 is some other  modality with a random feature dimension
    compseq_2_data = {}
    compseq_2_feature_dim = numpy.random.randint(low=20, high=100, size=1)
    random_init(compseq_2_data, compseq_2_feature_dim)

    compseq_2 = computational_sequence("my_compseq_2")
    compseq_2.setData(compseq_2_data, "my_compseq_2")

    # NOTE: if you don't want to manually input the metdata, set it by creating a metdata key-value dictionary based on mmsdk/mmdatasdk/configurations/metadataconfigs.py

    meta_dict = {
        "root name": "test data",  # name of the featureset
        "computational sequence description": "this is a dataset description",  # name of the featureset
        "dimension names": None,
        "computational sequence version": "1.0",  # the version of featureset
        "alignment compatible": "",  # name of the featureset
        "dataset name": "Test",  # featureset belongs to which dataset
        "dataset version": "1.0",  # the version of dataset
        "creator": "D. Chan",  # the author of the featureset
        "contact": "xxx@uibe.edu.cn",  # the contact of the featureset
        "featureset bib citation": "",  # citation for the paper related to this featureset
        "dataset bib citation": ""  # citation for the dataset
    }

    compseq_1.deploy("compseq_1.csd", my_metadata=meta_dict)
    compseq_2.deploy("compseq_2.csd", my_metadata=meta_dict)

    # now creating a toy dataset from the toy compseqs
    mydataset_recipe = {"compseq_1": "compseq_1.csd", "compseq_2": "compseq_2.csd"}
    mydataset = mmdataset(mydataset_recipe)
    # let's also see if we can align to compseq_1
    mydataset.align("compseq_1")