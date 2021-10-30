import os

root_path=r"examples\medmnist_data"

# start configure
flags=[f for f in os.listdir(root_path)]
print(flags)

for flag in flags:
    os.system(f"python step1_create_med_feat_lib.py {flag}")