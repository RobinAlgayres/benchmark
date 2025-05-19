import os,sys
import random
import numpy as np
from format_text import space_characters
from eval_pairs import read_pretraining_data

def make_bad_reflexive(sentence):
    sentence=sentence.split(' ')
    wrong_reflexives={'herself':['themselves','itself','ourselves'],'himself':['themselves','itself','ourselves'],'ourselves':['himself','itself','herself','themselves'],'themselves':['himself','herself','itself','ourselves'],'itself':['himself','herself','themselves','ourselves']}
    sentences=[]
    for i in range(len(sentence)):
        word=sentence[i]
        if word in wrong_reflexives:
            for choice in wrong_reflexives[word]:
                new_sentence=sentence
                new_sentence[i]=choice
                sentences.append(' '.join(new_sentence))
    #print(sentence)
    #print(sentences)
    return sentences

def make_bad_possessive(sentence):
    sentence=sentence.split(' ')
    wrong_possessive={'hers':['their','its','ours'],'his':['their','its','ours'],'ours':['his','its','hers','their'],'their':['his','its','hers','ours'],'its':['his','ours','hers','their']}
    wrong_pronouns={'him':['them','it'],'her':['them','it'],'them':['it','her','him'],'it':['them','him','her']}
    sentences=[]
    for i in range(len(sentence)):
        word=sentence[i]
        if word in wrong_possessive:
            for choice in wrong_possessive[word]:
                new_sentence=sentence
                new_sentence[i]=choice
                sentences.append(' '.join(new_sentence))
        elif word in wrong_pronouns:
            for choice in wrong_pronouns[word]:
                new_sentence=sentence
                new_sentence[i]=choice
                sentences.append(' '.join(new_sentence))
    return sentences

def format_sentence(sentence,word,dictionnary=None):
    if sentence[0]!='[' or sentence[-1]!=']':
        return None,None,None 
    sentence=sentence[1:-1]
    sentence=list(filter(None, sentence.split(' ')))
    formatted_generation=[]
    unknown_word=False
    for genword in sentence:
        genword,_,_,_=space_characters(genword,map_letters)
        if genword is None:
            unknown_word=True
            break
        genword=genword.split(' ')
        for sub_genword in genword:
            if dictionnary is not None and sub_genword not in dictionnary:
                unknown_word=True
                break
        formatted_generation+=genword 
    #print(formatted_generation)
    if word not in formatted_generation:
        return None,unknown_word,None
    index=formatted_generation.index(word)
    formatted_generation=' '.join(formatted_generation)
    return formatted_generation, index , unknown_word

map_letters={}
map_letters['foreigns']=[]
map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
map_letters['symbols']='!"$%&\'()*,-.0123456789:;?@[]'
map_letters['all']=map_letters['letters']+map_letters['symbols']


corpus='100M'
#frequency_file='BabyLM_2024_formatted/blimp_pairs_'+corpus+'_generations_morebins.txt'
#word_pair_file='BabyLM_2024_formatted/feasable_blimp_pairs_'+corpus+'_morebins'
frequency_file='BabyLM_2024_formatted/compositionality_task_'+corpus+'_filtered_generations_agreements.txt'
word_pair_file='BabyLM_2024_formatted/compositionality_task_'+corpus+'_agreements'
word_pair_file2='BabyLM_2024_formatted/compositionality_task_'+corpus+'_agreements'
pretraining_file='BabyLM_2024_formatted/longtail_'+corpus
dictionnary_file='BabyLM_2024_formatted/dictionnary_'+corpus
dictionnary=set()
if os.path.isfile(word_pair_file):
    assert False,(word_pair_file,'already exist')
if os.path.isfile(word_pair_file2):
    assert False,(word_pair_file2,'already exist')



with open(dictionnary_file) as buf:
    lines=buf.readlines()
    for line in lines:
        dictionnary.add(line.rstrip())
c=0
freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
output,output2=[],[]
with open(frequency_file) as buf:
    lines=buf.readlines()
context_data=read_pretraining_data(pretraining_file)

for line in lines:
    line=line.rstrip().split('|')
    if len(line)==6:
        bin,w1,w2,pos,g1,g2=line
    else:
        bin,w1,w2,pos,_,g1,g2=line
        dictionnary=None
    g1,ig1,unknown_w1=format_sentence(g1,w1,dictionnary)
    g2,ig2,unknown_w2=format_sentence(g2,w2,dictionnary)
    
    if unknown_w1 or g1 is None:
       #print(unknown_w1,g1)
        continue
    if unknown_w2 or g2 is None:
        #print(unknown_w2,g2)
        continue
    if len(g1.split(' '))<3 or len(g2.split(' '))<3:
        continue

    f1=context_data[w1]['freq']
    f2=context_data[w2]['freq']
    b1=np.where(f1>=freq_bins)[0][-1]
    b2=np.where(f2>=freq_bins)[0][-1]
    bin=str(min(b1,b2))
    s1,s2,i1,i2='s1','s2','0','0'
    if w1!=w2:
        output.append('|'.join((bin,pos,w1,s1,i1,g1,str(ig1),w2,s2,i2,g2,str(ig2))))
    else:
        print(g1,g2)
        wrong_g1s=make_bad_reflexive(g1)
        wrong_g2s=make_bad_possessive(g2)
        for wrong_g1 in wrong_g1s:
            output2.append('|'.join((g1,wrong_g1,bin,'reflexive')))
        for wrong_g2 in wrong_g2s:
            output2.append('|'.join((g2,wrong_g2,bin,'possessive')))
    c+=1
print('total correct sentences',c,len(lines))


with open(word_pair_file,'w') as buf:
    buf.write('\n'.join(output)+'\n')
with open(word_pair_file2,'w') as buf:
    buf.write('\n'.join(output2)+'\n')

    