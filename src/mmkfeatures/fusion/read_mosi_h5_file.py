import h5py

file_name="../full_examples/cmumosi/" \
          "http___immortal.multicomp.cs.cmu.edu_CMU-MOSI_language_CMU_MOSI_TimestampedPhones.csd"

file_name="../../../examples/compseq_1.csd"

import nexusformat.nexus as nx
f = nx.nxload(file_name)
print(f.tree)

print()
with h5py.File(file_name, "r") as f:
    for key in f.keys():
        print(key)  # Names of the groups in HDF5 file.

    # Get the HDF5 group
    group = f[key]
    # Checkout what keys are inside that group.
    for key in group.keys():
        print(key)

    print(list(group["metadata"]))
    meta_keys=list(group["metadata"])
    for mk in meta_keys:
        print("meta key: ",mk)
        print("-> ",group["metadata"][mk][0])

    print()
    content_ids=list(group["data"])
    print(content_ids)
    for k in content_ids: # each video
        print("[Content ID] -> ",k)
        part=group["data"][k]
        # print(list(part))
        features=part['features']
        intervals=part['intervals']
        # print(features)
        # print(intervals)
        print("[features] ->")
        for f in features[:5]:
            print("--> ",f)
        print("number of features: ",len(features))
        print("[intervals] ->")
        for inter in intervals[:5]:
            print("--> ",inter)
        print("number of intervals: ",len(intervals))
        print()









