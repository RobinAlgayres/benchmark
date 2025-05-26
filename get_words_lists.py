import numpy as np
import os,sys
from preprocessing_utils import format_word, space_characters, format_pos
from spellchecker import SpellChecker
import multiprocessing
import nltk
import json

def update_dict(line,map_letters,spell,char_dict):
    line=line.strip()
    line=line.replace('\t',' ')
    sentences=line.split('.')
    for sentence in sentences:
        sentence=sentence.split(' ')
        if len(sentence)==0:
            continue
        pos_sentence=nltk.pos_tag(sentence)
        for i in range(len(pos_sentence)):
            form,pos=pos_sentence[i]
            if len(spell.known([form]))==0 or pos in ['NNP','NNPS']:
                #the word form must belong to the english dict
                #also removing here most of the named entities
                continue
            if len(form)==0:
                continue 
            pos=format_pos(form,pos)
            #now getting rid of upper case information
            form=form.lower() 
            form=space_characters(form,map_letters)
            if form is None:
                #some word contains illegal characters and will be skipped
                continue
            #form can now contain severl white space
            for word in form.split(' '):
                if len(word)<=2:
                    continue
                if len(spell.known([word]))==0:
                    #the word form must belong to the english dict
                    continue    
                if word not in char_dict:
                    char_dict[word]={'freq':0,'POS':{}}
                char_dict[word]['freq']+=1
                if pos not in char_dict[word]['POS']:
                    char_dict[word]['POS'][pos]=0
                char_dict[word]['POS'][pos]+=1

def get_word_list(args):
    path,fid,output_char_dir=args
    print(fid)
    c=0
    char_dict={}
    spell = SpellChecker()
    map_letters={}
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['accepted_chars']='abcdefghijklmnopqrstuvwxyz!"$%&\'()*,-.0123456789:;?@[]'
    with open(path) as buf:
        for line in buf:
            #adding in dict all words that belong to the English dictionnary
            #if symbols are around the word, we may either skip the word
            #or separate this word from the symbols
            update_dict(line,map_letters,spell,char_dict)
            #if c>200:
            #    break
            c+=1
    output_file=os.path.join(output_char_dir,fid)
    with open(output_file,'w') as buf:
        buf.write(json.dumps(char_dict))
    print('saving',output_file,'with vocabulary size:',len(char_dict))

if __name__=='__main__':
    #list all words, computing their frequency and their frequency per POS
    #each word is checked independantly and separated by white space from neighboring symbols
    #some words are rejected altogether if contain illegal characters
    ncpus=5
    data='../shared/data/BabyLM_2024/text_data/train_100M/'
    #path to output dataset, will be created
    output_char_dir='tasks_100M/wordlist_per_file/'
    if not os.path.isdir(output_char_dir):
        os.makedirs(output_char_dir)

    #get word list and POS for each fid.
    arguments=[]
    for fid in os.listdir(data):
        path=os.path.join(data,fid)
        arguments.append((path,fid,output_char_dir))

    if ncpus==1:
        for argument in arguments:
            get_word_list(argument)
    else:
        with multiprocessing.Pool(processes=ncpus) as pool:
            pool.map(get_word_list, arguments) 
   
