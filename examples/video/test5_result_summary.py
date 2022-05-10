import os

lines=open("results.txt","r",encoding="utf-8")

list_x=[]
dict_y={}


for idx,line in enumerate(lines):
    if line.strip()=="":
        continue
    if idx==0:
        continue
    fs=line.strip().split("\t")
    print(fs)
    if fs[0] not in list_x:
        list_x.append(fs[0])
    if fs[1] not in dict_y:
        dict_y[fs[1]]=[fs[2]]
    else:
        dict_y[fs[1]].append(fs[2])

list_y=list(dict_y.keys())
print("")
print("size"+"\t".join(list_y))
for idx,x in enumerate(list_x):
    line=x+"\t"
    for y_label in list_y:
        line+=str(dict_y[y_label][idx])+"\t"
    print(line)
