## Basic Computational Sequence

This base class is deprived from the open-source CMU-Multimodal-SDK project which allows us to store multimodal objects like audio and video files. The core features of the ```computational sequence``` in the SDK is to develop a simple way to store each chunk's features in an order in video/audio files. For example, we can divide a 1-minute video into 60 1-second clips which can be stored in a time order. Then each clip is represented by its extracted features. The computation sequence class considers all objects have a basic property which is time. 

In our project, we extend the concept of computational sequence in many way, specially providing a more common way to store, fuse and retrieve extracted features from all sources. In this section, we firstly describe the basic usage of the computational sequencece in our project. 

Here is a toy example to show the use of computational sequence. 

1. First, assume we have a list of keys for multimodal contents. Each content has a key. For example, in this case, we generate a series of keys adn their contents like: 

```python 
if __name__=="__main__":
	vid_keys=["video1","video2","video3","video4","video5","Hello","World","UG3sfZKtCQI"]
	
	#let's assume compseq_1 is some modality with a random feature dimension
	compseq_1_data={}
	compseq_1_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_1_data,compseq_1_feature_dim)
```

2. We define a `computational_sequence` class by specifying its name.

```python
    compseq_1=computational_sequence("my_compseq_1")
	compseq_1.setData(compseq_1_data,"my_compseq_1")
```

3. Let's create another computational_sequence object with some random content features.

```python 
    compseq_2_data={}
	compseq_2_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_2_data,compseq_2_feature_dim)

	compseq_2=computational_sequence("my_compseq_2")
	compseq_2.setData(compseq_2_data,"my_compseq_2")

```

4. Define the metadata of each computational_sequence object and deploy(save) to the local disk.

```python
    compseq_1.deploy("compseq_1.csd",my_metadata=meta_dict)
	compseq_2.deploy("compseq_2.csd",my_metadata=meta_dict)
```

5. We can merge different computational_sequence objects into one sigle object by applying: 

```python
    # now creating a toy dataset from the toy compseqs
	mydataset_recipe={"compseq_1":"compseq_1.csd","compseq_2":"compseq_2.csd"}
	mydataset=mmdataset(mydataset_recipe)
```

6. Finally, we can try to align the features of one file to another by applying:

```python
    mydataset.align("compseq_1")
```

A complete example of showing basic usage of `computational_sequence` is below:
```python
from mmkfeatures.fusion.computational_sequence import computational_sequence
from mmkfeatures.fusion.dataset import mmdataset
import numpy

def random_init(compseq,feat_dim):
	for vid_key in vid_keys:
		num_entries=numpy.random.randint(low=5,high=100,size=1).astype(int)[0]
		feat_dim1=feat_dim.astype(int)[0]
		print("num_entries:",num_entries)
		print("fea_dim:",feat_dim)
		compseq[vid_key]={}
		compseq[vid_key]["features"]=numpy.random.uniform(low=0,high=1,size=[num_entries,feat_dim1])
		#let's assume each video is one minute, hence 60 seconds. 
		compseq[vid_key]["intervals"]=numpy.arange(start=0,stop=60+0.000001,step=60./((2*num_entries)-1)).reshape([num_entries,2])

if __name__=="__main__":
	vid_keys=["video1","video2","video3","video4","video5","Hello","World","UG3sfZKtCQI"]
	
	#let's assume compseq_1 is some modality with a random feature dimension
	compseq_1_data={}
	compseq_1_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_1_data,compseq_1_feature_dim)

	compseq_1=computational_sequence("my_compseq_1")
	compseq_1.setData(compseq_1_data,"my_compseq_1")

	#let's assume compseq_2 is some other  modality with a random feature dimension
	compseq_2_data={}
	compseq_2_feature_dim=numpy.random.randint(low=20,high=100,size=1)
	random_init(compseq_2_data,compseq_2_feature_dim)

	compseq_2=computational_sequence("my_compseq_2")
	compseq_2.setData(compseq_2_data,"my_compseq_2")

	#NOTE: if you don't want to manually input the metdata, set it by creating a metdata key-value dictionary based on mmsdk/mmdatasdk/configurations/metadataconfigs.py

	meta_dict={
		"root name":"test data",  # name of the featureset
		"computational sequence description":"this is a dataset description",  # name of the featureset
		"dimension names":None,
		"computational sequence version":"1.0",  # the version of featureset
		"alignment compatible":"",  # name of the featureset
		"dataset name":"Test",  # featureset belongs to which dataset
		"dataset version":"1.0",  # the version of dataset
		"creator":"D. Chan",  # the author of the featureset
		"contact":"xxx@uibe.edu.cn",  # the contact of the featureset
		"featureset bib citation":"",  # citation for the paper related to this featureset
		"dataset bib citation":"" # citation for the dataset
	}

	compseq_1.deploy("compseq_1.csd",my_metadata=meta_dict)
	compseq_2.deploy("compseq_2.csd",my_metadata=meta_dict)

	#now creating a toy dataset from the toy compseqs
	mydataset_recipe={"compseq_1":"compseq_1.csd","compseq_2":"compseq_2.csd"}
	mydataset=mmdataset(mydataset_recipe)
	#let's also see if we can align to compseq_1
	mydataset.align("compseq_1")
```

The above example is a simple toy one and not suitable for complicated multimodal features use. Therefore, based on the `computational sequence`, we developed a brand-new and complicated one named `computatoinal_sequencex` to facilicate a common frame of storing and manipulating multimodal features for high-level applications in many fields. We will discuss the new one in other section. 