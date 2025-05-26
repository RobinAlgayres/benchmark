import numpy as np
import os,sys
import ast
from preprocessing_utils import format_word, space_characters, format_pos
from spellchecker import SpellChecker
import tqdm
import json

def concat_dict(path,char_dict):
    json.load
    with open(path) as buf:
        words=json.load(buf)
        for word in words:
            h=words[word]
            if word not in char_dict:
                char_dict[word]={'freq':0,'POS':{}}
            char_dict[word]['freq']+=h['freq']
            for pos in h['POS']:
                if pos not in char_dict[word]['POS']:
                    char_dict[word]['POS'][pos]=0
                char_dict[word]['POS'][pos]+=h['POS'][pos]
                

if __name__=='__main__':
    data='tasks_10M/wordlist_per_file/'
    output_file_morpho='tasks_10M/longtail_morpho'    
    output_file_infl='tasks_10M/longtail_infl'   
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    char_dict={}
    words_per_bins={}
    longtail_morpho,longtail_infl=[],[]
    for fid in os.listdir(data):
        path=os.path.join(data,fid)
        concat_dict(path,char_dict)
    print('nb keys',len(char_dict.keys()))
    spell = SpellChecker()
    i=0
    j=0
    for form in tqdm.tqdm(char_dict):
        # A cluster is the set of all inflection+POS that comes from one base word form
        # Checking that among all POS tags and all inflections for that word, the frequency of that word 
        # with that POS tag is roughly the same as the total frequency of its cluster. 
        # intuitively, it means this word is particularly frequent in that inflection and POS tag.
       
        most_common_pos_freq=0
        most_common_pos='UNK'
        #getting most frequent pos tag: noun,verb or UNK 
        for pos in char_dict[form]['POS']:
            pos_freq=char_dict[form]['POS'][pos]
            if pos_freq>most_common_pos_freq:
                most_common_pos_freq=pos_freq
                most_common_pos=pos
        if most_common_pos=='UNK':
            #keeping only nouns and verbs
            i+=1
            #print(i)
            continue
        #get inflections for this noun or verb
        inflections=format_word(form,most_common_pos,spell)
        if len(inflections)==0:
            #if word is too short it will not have any inflection
            continue

        cluster_freq=char_dict[form]['POS'][most_common_pos] #sum of freq for all POS of that word
        cluster=[form,str(most_common_pos_freq)]
        most_common_pos_freq_bin=np.where(most_common_pos_freq>=freq_bins)[0][-1]
        for inflection,infl_pos in inflections:
            inflection_freq=0
            if inflection==form:
                continue
            if inflection in char_dict and infl_pos in char_dict[inflection]['POS']:
                inflection_freq=char_dict[inflection]['POS'][infl_pos]#adding the freq for all POS of that inflection
            
            cluster.extend([inflection,str(inflection_freq)])
            cluster_freq+=inflection_freq

            inflection_freq_bin=np.where(inflection_freq>=freq_bins)[0][-1]
            if abs(inflection_freq_bin-most_common_pos_freq_bin)<2:
                longtail_infl.append('|'.join((str(most_common_pos_freq_bin),form,most_common_pos,form+' '+str(most_common_pos_freq)+' '+inflection+' '+str(inflection_freq))))
        if len(cluster)>2: #we have added at least one inflection to the cluster
            cluster_freq_bin=np.where(cluster_freq>=freq_bins)[0][-1]
            #checking that the freq for that word/POS is in the same bin as the sum
            #of frequencies of this word's inflections 
            if cluster_freq_bin!=most_common_pos_freq_bin:
                continue
        
        key=str(cluster_freq_bin)+'_'+most_common_pos.split('_')[0]
        if key not in words_per_bins:
            words_per_bins[key]=0
        if words_per_bins[key]>=4000:
            #if more than 4k words in that bin and POS, skipping
            continue
        words_per_bins[key]+=1 
        cluster=' '.join(cluster)
        longtail_morpho.append('|'.join((str(most_common_pos_freq_bin),form,most_common_pos,cluster)))
      
    for key in words_per_bins:
        print(key,words_per_bins[key])
    with open(output_file_morpho,'w') as buf:
        buf.write('\n'.join(longtail_morpho)+'\n')
    with open(output_file_infl,'w') as buf:
        buf.write('\n'.join(longtail_infl)+'\n')
