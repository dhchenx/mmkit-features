from quickcsv import *
import os
list_all=[]
for folder in os.listdir('text_c10'):
    for file in os.listdir(os.path.join('text_c10',folder)):
        text=read_text(os.path.join('text_c10',folder,file))
        # print(text)
        list_all.append({
            "text":text
        })

write_csv('data3.csv',list_all)