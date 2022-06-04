import os
from quickcsv import *

root_path="datasets/CUB_200_2011/text_c10/"

all_text=""
for file in os.listdir(root_path):
    for file1 in os.listdir(f"{root_path}/{file}"):
        path=f"{root_path}/{file}/{file1}"
        text=read_text(path)
        all_text+=text+"\n"

write_text("birds.txt",all_text)