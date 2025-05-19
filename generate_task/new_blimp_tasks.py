import numpy as np
import random, json
import tqdm
import os, ast
from eval_pairs import get_context_util, get_context_util_nopos, read_pretraining_data, format_sentence
from easy_inflections import format_word

if __name__ == '__main__':
    #find all inflections that never appear in the training set.
    cuda=True
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    corpus='10M'
    cluster_file='BabyLM_2024_formatted/freq_bins_'+corpus+'_512'
    pretraining_file='BabyLM_2024_formatted/longtail_'+corpus
    output_file='BabyLM_2024_formatted/infl_pairs_tobefiltered_'+corpus
    print(pretraining_file)
    context_data=read_pretraining_data(pretraining_file)
    with open(cluster_file) as buf:
        lines=buf.readlines()
    out=[]
    pairs={}
    success={}
    for i in range(len(freq_bins)):   
        success[i]={}
        for pos in ['VERB','NOUN','ADJ','UNK']:
            if pos not in success[i]:
                success[i][pos]=0

    for line in tqdm.tqdm(lines):
        bin,word,pos,_,_,_,sentence=line.rstrip().split('/')
        base_pos=pos.split('_')[0]
        for inflection in format_word(word,base_pos):
            if inflection==word:
                continue 
            #a pair can only be added once
            key=[word,inflection]
            key.sort()
            key='-'.join(key)
            if key in pairs:
                continue
            pairs[key]=0
            infl_freq=context_data[inflection]['freq']
            success[int(bin)][base_pos]+=1
            infl_bin=np.where(infl_freq>=freq_bins)[0][-1]
            bin=str(max(infl_bin,int(bin)))
            lower_pos=base_pos.lower()
            if lower_pos=='adj':
                lower_pos='adjective'
            prompt1=' '.join(("Given the two",lower_pos+'s',"\'",word,"\' and \'",inflection,"\'. Can you tell if they are two inflections of the same",lower_pos,"? Answer by yes or no. Write your answer in between brackets."))
            if lower_pos=='noun':
                prompt2=' '.join(("Given the",lower_pos,"\'",word,"\'. Can this noun take a reflexive pronoun? Answer by yes or no. Write your answer in between brackets."))
            else:
                prompt2='empty'
            #out.append('|'.join((bin,word,inflection,base_pos,sentence,prompt1,prompt2)))
            out.append('|'.join((bin,word,inflection,base_pos,sentence,prompt1)))
for bin in success:
    print(success[bin])
with open(output_file,'w') as buf:
    buf.write('\n'.join(out)+'\n')