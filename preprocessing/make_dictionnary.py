import os,sys,re
import tqdm
from format_text import space_characters
dataset_file=sys.argv[1]#BabyLM_2024_formatted/all_10M_raw
dictionnary_file=sys.argv[2]
map_letters={}
map_letters['foreigns']=[]
map_letters['letters']='abcdefghijklmnopqrstuvwxyz0123456789'

d={}
with open(dataset_file) as buf:
    lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        for word in line.rstrip().split(' '):
            word=word.lower()
            if len(word)==0:
                continue
            word,_,_=space_characters(word,map_letters)
            for subword in word.split(' '):
                if subword==' ' or len(subword)==0:
                    continue
                if subword not in d:
                    d[subword]=0
                d[subword]+=1

output=[]
d=dict(sorted(d.items(), key=lambda item: item[1]))
for key in d:
    output.append(key+' '+str(d[key]))

#with open(dictionnary_file,'w') as buf:
#    buf.write('\n'.join(output)+'\n')