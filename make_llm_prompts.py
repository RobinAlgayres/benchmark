import numpy as np
import random, json
import tqdm
import os, ast
from utils import get_context_util, get_context_util_nopos, read_pretraining_data, format_sentence
from easy_inflections import format_word

def make_prompt_minimal(word,inflection,pos,rule):
   return prompt

if __name__ == '__main__':
    #find all inflections that never appear in the training set.
    cuda=True
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    corpus='100M'
    cluster_file='BabyLM_2024_formatted/freq_bins_'+corpus
    output_file='BabyLM_2024_formatted/prompts_'+corpus+'_'
    dictionnary_file='BabyLM_2024_formatted/dictionnary_'+corpus
    dictionnary={}
    with open(dictionnary_file) as buf:
        lines=buf.readlines()
        for line in lines:
            word,freq=line.rstrip().split(' ')
            assert word not in dictionnary
            dictionnary[word]=int(freq)
    with open(cluster_file) as buf:
        lines=buf.readlines()
    out=[]
    pairs={}
    success={}
    for i in range(len(freq_bins)):   
        success[i]={}
        for pos in ['verb','noun','adjective','unk']:
            if pos not in success[i]:
                success[i][pos]=0

    used_inflections=set()
    used_words=set()
    for line in tqdm.tqdm(lines):
        _,word,pos,_,_,_,sentence=line.rstrip().split('|')
        base_pos=pos.split('_')[0]
        inflections=format_word(word,base_pos)
        base_pos=base_pos.lower() 
        if base_pos=='adj':
            base_pos='adjective'
        if word in used_words:
            #guaranteing that each word form is used once as a baseword
            continue

        f=dictionnary.get(word,0)
        assert f>0,(word,line)
        bin=str(np.where(f>=freq_bins)[0][-1])
        
        prompt=' '.join(("Given the",base_pos,"\'",word,"\'. Can you write a simple sentence that contains the",base_pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets."))
        out.append('|'.join((bin,word,'BASEWORD',pos,sentence,prompt)))
        success[int(bin)][base_pos]+=1
        used_words.add(word)
        
        for inflection in inflections:
            if inflection==word:
                continue 
            #a pair can only be added once
            key=[base_pos,word,inflection]
            key.sort()
            key='-'.join(key)
            if key in pairs:
                continue
            pairs[key]=0
            infl_freq=dictionnary.get(inflection,0)
            infl_bin=np.where(infl_freq>=freq_bins)[0][-1]
            infl_bin=str(min(infl_bin,int(bin)))

            prompt=' '.join(("Given the two",base_pos+'s \''+word+'\' and \''+inflection+'\'',". Can you tell if they are two inflections of the same",base_pos,"? Answer by yes or no. Write your answer in between brackets."))
            out.append('|'.join((infl_bin,word,'AREINFLECTIONS',pos,inflection,prompt)))
            
            #prompt=' '.join(("Given the",base_pos,"\'",inflection,"\'. Can you write a simple sentence that contains the",base_pos,"\'",inflection,"\' using at least 20 words. Make it simple. Write only this sentence between brackets."))
            prompt=' '.join(("Please write a minimal pair of sentences using the",base_pos+'s \''+word+'\' and \''+inflection+'\'',"in similar contexts. Enclose both sentences within brackets."))
            out.append('|'.join((infl_bin,word,'INFLECTION',pos,inflection,prompt)))
            out.append('|'.join((infl_bin,word,'INFLECTION',pos,inflection,prompt)))
            out.append('|'.join((infl_bin,word,'INFLECTION',pos,inflection,prompt)))
            out.append('|'.join((infl_bin,word,'INFLECTION',pos,inflection,prompt)))
                
        #if len(out)>100:
        #    break    
for bin in success:
    print(success[bin])
with open(output_file,'w') as buf:
    buf.write('\n'.join(out)+'\n')