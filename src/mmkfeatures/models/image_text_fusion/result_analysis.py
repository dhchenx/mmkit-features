
lines=open(r'C:\Users\dell\Desktop\UMKG\results\conv-model.txt','r',encoding='utf-8')

print(f'Epoch\tLoss')
for l in lines:
    l=l.strip()
    if l.startswith("Epoch:"):
        fs=l.split(" 	Training Loss: ")
        epoch=int(fs[0].split(":")[1].strip())
        loss=float(fs[1].strip())
        print(f'{epoch}\t{loss}')

