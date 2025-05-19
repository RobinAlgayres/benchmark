import os,sys,tqdm
import random
import numpy as np
from format_text import space_characters
from eval_pairs import read_pretraining_data

def find_two_generations(word,inflection,g):
    assert False, 'TODO: this should output the two sentences created by the LLMs'
    g=g.replace('\\','')
    g=g.replace('\"','')
    g=g.replace('\'','')   
    #removing empty space
    g=' '.join(list(filter(None, g.split(' '))))

    #the pattern that enables to split the sentence is not always a period.
    for pattern in ['.','!','?','/',', but',', while',', whereas',', and ',',',';']:
        #final period is not a separator
        if pattern in g[:-1]:
            ind=g.find(pattern)
            g2=g[ind+len(pattern)+1:]
            g1=g[:ind]
            #if pattern==', while':
                #print(g1,'|',g2,'|',g)
            if len(g1)<3 or len(g2)<3:
                continue
            else:
                break

            
    if len(g1)==0 or len(g2)==0:
        print('G1:',g1)
        print('G2:',g2)
    
    g1,ig1,unknown_w1=format_sentence(g1,w1,dictionnary)
    if unknown_w1:
        #the LLM may use w1 in g2 instead of g1.
        tmp=w2
        w2=w1
        w1=tmp
        g1,ig1,unknown_w1=format_sentence(g1,w1,dictionnary)
        if unknown_w1:
            return
    g2,ig2,unknown_w2=format_sentence(g2,w2,dictionnary)
    if unknown_w2:
        return
    indg=generation.find(word)
    if indg==-1 or generation[indg+len(word)] in 'abcdefghijklmnopqrstuvwxyz' or generation[indg-1]!=' ':
         #print(inflection,word,'not in',generation)
        return
    indg,inds=str(indg),str(inds)
    return w1,ig1,g1,w2,ig2,g2

def find_unknown_word(generation,map_letters,dictionnary,word):
    unknown_word=None
    for genword in generation.split(' '):
        if len(genword)==0:
            continue
        genword,_,_=space_characters(genword,map_letters)
        if genword is None:
            unknown_word=genword
            break
        for sub_genword in genword.split(' '):
            #rejecting sentences that contains unknown words
            #yet the target word might be unknown (inflection of known word)
            if sub_genword.lower()!=word and sub_genword.lower() not in dictionnary:
                unknown_word=sub_genword
                break
    return unknown_word

if __name__ == '__main__':
    corpus='100M'
    #input
    pretraining_file='BabyLM_2024_formatted/longtail_'+corpus
    frequency_file='BabyLM_2024_formatted/generations_'+corpus+'.txt'
    dictionnary_file='BabyLM_2024_formatted/dictionnary_'+corpus
    #output
    wordswap_file='BabyLM_2024_formatted/wordswap_pairs_'+corpus
    inflswap_file='BabyLM_2024_formatted/inflswap_pairs_'+corpus
    
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    map_letters={}
    map_letters['foreigns']=[]
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'

    max_pairs_per_pos=1000
    dictionnary={}
    with open(dictionnary_file) as buf:
        lines=buf.readlines()
        for line in lines:
            word,freq=line.rstrip().split(' ')
            assert word not in dictionnary
            dictionnary[word]=int(freq)

    c=0
    are_inflections,words,inflections=set(),{},{}
    with open(frequency_file) as buf:
        lines=buf.readlines()
    seen_words=set() #for some reason some base words are duplicated
    for line in tqdm.tqdm(lines):
        line=line.rstrip().split('|')
        if len(line)!=7:
            continue
        bin,word,metadata,pos,data,prompt,generation=line
        if metadata!='BASEWORD':
            continue
        #formatting the generated sentence from the llm
        start=generation.rfind('[')
        end=generation.rfind(']')
        if start==-1 or end==-1:
            continue
        
        generation=generation[start+1:end]

        if metadata=='AREINFLECTIONS':
            inflection=data
            #the llm was asked if word and inflection are indeed two inflection of the same word.
            assert word in dictionnary,(word,inflection)
            if generation.lower()=='yes':
                key=[word,inflection]
                key.sort()
                key='-'.join(key)
                if key not in are_inflections:
                    are_inflections[key]=[]
                are_inflections.add(key)
        elif metadata=='BASEWORD':
            if word in seen_words:
                continue
            seen_words.add(word)
            #checking if an unknown word has been used
            unknown_word=find_unknown_word(generation,map_letters,dictionnary,word)
            if unknown_word is not None:
                continue
            #'word' will be in the sentence only if it is a baseword
            #sometimes the word is not found because the LLM has put a capital letter to the target
            #word, this is good it helps remove named entities 
            sentence=data+' '
            inds=sentence.find(word)
            #the next letter must not be another letter
            #print(inds,word,sentence,len(sentence))
            if inds==-1 or sentence[inds+len(word)] in 'abcdefghijklmnopqrstuvwxyz' or sentence[inds-1]!=' ':
                #print(inflection,word,'not in',sentence,inds)
                continue
            indg=generation.find(word)
            if indg==-1 or generation[indg+len(word)] in 'abcdefghijklmnopqrstuvwxyz' or generation[indg-1]!=' ':
                #print(inflection,word,'not in',generation)
                continue
            indg,inds=str(indg),str(inds)
            
            if bin not in words:
                words[bin]={}
            if pos not in words[bin]:
                words[bin][pos]=[]
            words[bin][pos].append((word,inds,sentence,indg,generation))

        elif metadata=='INFLECTION':
            w1,ig1,g1,w2,ig2,g2=find_two_generations(word,inflection,generation)
            unknown_word=find_unknown_word(g1,map_letters,dictionnary,w1)
            if unknown_word is not None:
                continue
            unknown_word=find_unknown_word(g2,map_letters,dictionnary,w2)
            if unknown_word is not None:
                continue
            if bin not in inflections:
                inflections[bin]={}
            if pos not in inflections[bin]:
                inflections[bin][pos]=[]
            inflections[bin][pos].append((w1,w2,pos,ig1,g1,ig2,g2))

wordswap_list,inflswap_list=[],[]

wordpairs=set()
inflpairs=set()
for bin in words:
    for pos in words[bin]:
        tmp=[]
        for i in range(len(words[bin][pos])-1): 
            w1,is1,s1,ig1,g1=words[bin][pos][i]
            for j in range(i+1,len(words[bin][pos])):
                
                w2,is2,s2,ig2,g2=words[bin][pos][j]
                if w1==w2:
                    continue
                #if new pair add it to the output
                key=[w1,w2]
                key.sort()
                key='-'.join((w1,w2))
                if key in wordpairs:
                    continue
                wordpairs.add(key)
                tmp.append('|'.join((str(bin),pos,w1,s1,is1,g1,ig1,w2,s2,is2,g2,ig2)))
        random.shuffle(tmp)
        print(bin,pos,len(tmp))
        wordswap_list+=tmp[:max_pairs_per_pos]

for bin in inflections:
    for pos in inflections[bin]:
        tmp_infl=[]
        for i in range(len(words[bin][pos])-1): 
            w1,ig1,g1,w2,ig2,g2=inflections[bin][pos][i]
            key=[w1,w2]
            key.sort()
            key='-'.join(key)
            if key not in are_inflections:
                print(key)
                continue
            base_pos=pos.split('_')[0]
            tmp_infl.append('|'.join((bin,base_pos,w1,g1,ig1,w2,g2,ig2)))
        random.shuffle(tmp_infl)
        inflswap_list+=tmp_infl[:max_pairs_per_pos]
        print(bin,pos,len(tmp_infl))

with open(wordswap_file,'w') as buf:
    buf.write('\n'.join(wordswap_list)+'\n')
with open(inflswap_file,'w') as buf:
    buf.write('\n'.join(inflswap_list)+'\n')