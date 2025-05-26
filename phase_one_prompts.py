import numpy as np
import random, json
import tqdm
import os, ast
from utils import get_context_util, get_context_util_nopos, read_pretraining_data, format_sentence
from easy_inflections import format_word


if __name__ == '__main__':
    corpus='100M'
    cluster_file='tasks_10M/longtail_morpho'
    cluster_file_infl='tasks_10M/longtail_infl'
    output_file='tasks_10M/phase_one_prompts'
    dictionnary={}
    out=[]
    with open(cluster_file) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        bin,word,pos=line.rstrip().split('|')[:3]
        base_pos=pos.split('_')[0].lower() 
        assert base_pos in ['noun','verb'],base_pos
        prompt=' '.join(("Given the",base_pos,"\'",word,"\'. Can you write a simple sentence that contains the",base_pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets."))
        out.append('|'.join((bin,word,'BASEWORD',pos,prompt)))
    
    with open(cluster_file_infl) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        bin,word,pos,_,inflection,pos_infl,_=line.rstrip().split('|')
        base_pos=pos.split('_')[0].lower() 
        prompt=' '.join(("Given the two",base_pos+'s \''+word+'\' and \''+inflection+'\'',". Can you tell if they are two inflections of the same",base_pos,"? Answer by yes or no. Write your answer in between brackets."))
        out.append('|'.join((bin,word,'AREINFLECTIONS',pos,inflection,prompt)))
        if base_pos=='noun':
            #verbs are harder to control for AgreementSwap, we remove them from here
            prompt=' '.join(("Given the two",base_pos+'s \''+word+'\' and \''+inflection+'\'',". Can this noun take a reflexive pronoun? Answer by yes or no. Write your answer in between brackets."))
            out.append('|'.join((bin,word,'ARESUBJECTS',pos,inflection,prompt)))

    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')