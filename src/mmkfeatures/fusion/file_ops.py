import sys
import h5py
import os
import json
from tqdm import tqdm
from mmkfeatures.fusion  import log
from mmkfeatures.fusion .metadataconfigs import *
from mmkfeatures.fusion.integrity_check import *


def read_CSD(resource, destination=None):
    if (resource is None): raise log.error("No resource specified for computational sequence!", error=True)
    if os.path.isfile(resource) is False:
        log.error("%s file not found, please check the path ..." % resource, error=True)
    try:
        h5handle = h5py.File('%s' % resource, 'r')
    except:
        raise log.error("%s resource is not a valid hdf5 computational sequence format ..." % resource, error=True)
    log.success("Computational sequence read from file %s ..." % resource)

    data=dict(h5handle[list(h5handle.keys())[0]]["data"])
    if "rel_data" in h5handle[list(h5handle.keys())[0]]:
        rel_data=dict(h5handle[list(h5handle.keys())[0]]["rel_data"])
    else:
        rel_data=None
    if "dicts" in h5handle[list(h5handle.keys())[0]]:
        dicts= dict(h5handle[list(h5handle.keys())[0]]["dicts"])
    else:
        dicts=None
    metadata=metadata_to_dict(h5handle[list(h5handle.keys())[0]]["metadata"])

    return h5handle, data,rel_data,dicts,metadata

# writing CSD files to disk
def write_CSD(data, rel_data,dicts, metadata, rootName, destination, compression, compression_opts, full_chunk_shape):
    log.status("Writing the <%s> computational sequence data to %s" % (rootName, destination))
    if compression is not None:
        log.advise("Compression with %s and opts -%d" % (compression, compression_opts))
    # opening the file
    writeh5Handle = h5py.File(destination, 'w')
    # creating the root handle
    rootHandle = writeh5Handle.create_group(rootName)

    # writing the data
    dataHandle = rootHandle.create_group("data")
    pbar = log.progress_bar(total=len(data.keys()), unit=" Contents", leave=False)
    for vid in data:
        vidHandle = dataHandle.create_group(vid)
        from mmkfeatures.fusion.mm_features_node import MMFeaturesNode
        import numpy
        temp=MMFeaturesNode("")
        if metadata["compressed"].lower()=="true":
            keys=temp.get_all_validated_keys_alias()
        else:
            keys=temp.get_all_validated_keys()
        if compression is not None:
            '''
            vidHandle.create_dataset("features", data=data[vid]["features"], compression=compression,
                                     compression_opts=compression_opts)
            vidHandle.create_dataset("intervals", data=data[vid]["intervals"], compression=compression,
                                     compression_opts=compression_opts)
            '''
            for k in keys:
                if k in data[vid].keys():
                    if k in ["features","intervals"] or k in ['F','ITV']:
                        vidHandle.create_dataset(k, data=data[vid][k],
                                                  compression=compression,
                                          compression_opts=compression_opts
                                                 )
                    else:
                        # print(k)
                        # print(data[vid][k])
                        if type(data[vid][k])==list:
                            #for x in data[vid][k]:
                            # data=numpy.array(data[vid][k],dtype='S4')
                            vidHandle.create_dataset(k, data=str(data[vid][k]))
                        elif type(data[vid][k])==dict:
                            vidHandle.create_dataset(k,data=str(data[vid][k]))
                        else:
                            vidHandle.create_dataset(k, data=data[vid][k])
        else:
            '''
            vidHandle.create_dataset("features", data=data[vid]["features"])
            vidHandle.create_dataset("intervals", data=data[vid]["intervals"])
            '''
            for k in keys:
                if k in data[vid].keys():
                    vidHandle.create_dataset(k, data=data[vid][k])


        pbar.update(1)
    pbar.close()

    pbar = log.progress_bar(total=len(rel_data.keys()), unit=" Relationships", leave=False)
    relDataHandler = rootHandle.create_group("rel_data")
    for rel_id in rel_data:
        relHandler = relDataHandler.create_group(rel_id)
        keys=rel_data[rel_id].keys()
        for k in keys:
            if type(rel_data[rel_id][k]) == list:
                relHandler.create_dataset(k, data=str(rel_data[rel_id][k]))
            elif type(rel_data[rel_id][k]) == dict:
                relHandler.create_dataset(k, data=str(rel_data[rel_id][k]))
            else:
                relHandler.create_dataset(k, data=rel_data[rel_id][k])
        pbar.update(1)
    pbar.close()

    pbar = log.progress_bar(total=len(rel_data.keys()), unit=" Dictionaries", leave=False)
    dictDataHandler = rootHandle.create_group("dicts")
    for dict_id in dicts:
        dictHandler = dictDataHandler.create_group(dict_id)
        keys = dicts[dict_id].keys()
        for k in keys:
            if type(dicts[dict_id][k]) == list:
                dictHandler.create_dataset(k, data=str(dicts[dict_id][k]))
            elif type(dicts[dict_id][k]) == dict:
                dictHandler.create_dataset(k, data=str(dicts[dict_id][k]))
            else:
                dictHandler.create_dataset(k, data=dicts[dict_id][k])
        pbar.update(1)
    pbar.close()

    log.success("<%s> computational sequence data successfully wrote to %s" % (rootName, destination))
    log.status("Writing the <%s> computational sequence metadata to %s" % (rootName, destination))
    # writing the metadata
    metadataHandle = rootHandle.create_group("metadata")
    for metadataKey in metadata.keys():
        metadataHandle.create_dataset(metadataKey, (1,), dtype=h5py.special_dtype(
            vlen=unicode) if sys.version_info.major is 2 else h5py.special_dtype(vlen=str))
        cast_operator = unicode if sys.version_info.major is 2 else str
        metadataHandle[metadataKey][0] = cast_operator(json.dumps(metadata[metadataKey]))

    writeh5Handle.close()

    log.success("<%s> computational sequence metadata successfully wrote to %s" % (rootName, destination))
    log.success("<%s> computational sequence successfully wrote to %s ..." % (rootName, destination))


def metadata_to_dict(metadata_):
    if (type(metadata_) is dict):
        return metadata_
    else:
        metadata = {}
        for key in metadata_.keys():
            try:
                metadata[key] = json.loads(metadata_[key][0])
            except:
                try:
                    metadata[key] = str(metadata_[key][0])
                except:
                    log.error("Metadata %s is in wrong format. Exiting ...!" % key)
        return metadata