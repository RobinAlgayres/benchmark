import numpy as np
import os,sys
from easy_inflections import format_word
from spellchecker import SpellChecker
import tqdm
import nltk
import time 

#take a text corpus where each line is prependded with the corpus name followed by a |.
#this script computes the list of words with their frequencies, pos tag(s) and corpus of origin.
def format_pos(word,pos,index=None):
    if word[0].isupper():
        if index is None:
            pos='UNK'
        elif pos in ['NNP','NNPS'] and index>0:
            pos='NE'
        else:
            pos='UNK' #superset of named entities to be sure to remove them all
    elif pos in ['NN', 'NNS']:    
        if word[-1]=='s':
            pos='NOUN_P'
        else:
            pos='NOUN'
    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        if word[-2:]=='ed':
            pos='VERB_Past'
        elif word[-3:]=='ing':
            pos='VERB_PresC'
        elif word[-1]=='s':
            pos='VERB_PresT'
        else:
            pos='VERB' 
    elif pos in ['JJ', 'JJR', 'JJS']:
        if word[-2:]=='er':
            pos='ADJ_R'
        elif word[-3:]=='est':
            pos='ADJ_S'
        else:
            pos='ADJ'
    else: #not dealing with adverbs
        pos='UNK'
    return pos

if __name__=='__main__':
    data=sys.argv[1]
    out_char=sys.argv[2]
    char_dict={}
    c=0

    several_pos=set()
    with open(data) as buf:
        lines=buf.readlines()
        for line in tqdm.tqdm(lines):
            line=line.strip().split('|')
            if len(line)!=2:
                continue
            corpus,sentences=line
            sentences=sentences.split('.')
            for sentence in sentences:
                sentence=sentence.split(' ')
                if len(sentence)==0:
                    continue
                nb_words=len(sentence)
                pos_sentence=nltk.pos_tag(sentence)
                
                for i in range(len(pos_sentence)):
                    form,pos=pos_sentence[i]
                    if len(form)==0:
                        continue
                    pos=format_pos(form,pos,i)
                    if pos=='NE':
                        continue
                    form=form.lower()
                    if form not in char_dict:
                        char_dict[form]={'word':form,'freq':0,'all_inflections':set(),'all_corpora':set(),'POS':{}}
                    char_dict[form]['freq']+=1
                    if pos not in char_dict[form]['POS']:
                        char_dict[form]['POS'][pos]={}
                        char_dict[form]['POS'][pos]['freq']=0
                        char_dict[form]['POS'][pos]['context']=None
                        char_dict[form]['POS'][pos]['max_len']=0
                        char_dict[form]['POS'][pos]['corpora']=set()
                        char_dict[form]['POS'][pos]['inflections']=[]

                    char_dict[form]['all_corpora'].add(corpus)
                    char_dict[form]['POS'][pos]['freq']+=1
                    char_dict[form]['POS'][pos]['corpora'].add(corpus)
                    #storing longest sentence for that POS
                    if nb_words>char_dict[form]['POS'][pos]['max_len']:
                        char_dict[form]['POS'][pos]['max_len']=nb_words
                        char_dict[form]['POS'][pos]['context']=' '.join(sentence)
                    #if len(char_dict[form]['POS'][pos]['context'])<10 and nb_words>10:
                        #context=' '.join(sentence[max(0,i-20):min(nb_words,i+20)])
            if False and c>200:
                break
            c+=1

    longtail=[]
    spell = SpellChecker()
    for form in tqdm.tqdm(char_dict):
        #form=formpos.split(' ')[0]
        if len(spell.known([form]))==0:
            #the word form must belong to the english dict
            #removing here most of the named entities
            continue
        #computing all inflections without constraint on number of letters
        char_dict[form]['all_inflections'].add(form)
        for pos in char_dict[form]['POS']:
            base_pos=pos.split('_')[0]
            if base_pos not in ['VERB','NOUN','ADJ']:
                continue
            inflections=format_word(form,base_pos)
            char_dict[form]['POS'][pos]['inflections']=inflections
            for inflection in inflections:
                char_dict[form]['all_inflections'].add(inflection)  

        if len(char_dict[form]['all_inflections'])==0:
            #some forms are too short and will produce no inflections (not even themselves)
            continue
        longtail.append(str(char_dict[form]))

    with open(out_char,'w') as buf:
        buf.write('\n'.join(longtail)+'\n')

