import os,sys,tqdm
import random, nltk
import numpy as np
from preprocessing_utils import check_capital_and_punc,find_reflexive,find_verb
import argparse

def make_prompt(s1,s2):
    prompt = (
    "Given the two sentences A and B:",
    "<start of sentence A> "+s1+" <end of sentence A>",
    "<start of sentence B> "+s2+" <end of sentence B>",
    "Which of the two sentences, A or B, is more physically accurate? Write your answer (A or B) in the brackets."
    )
    return ' '.join(prompt)


def find_index(sentence,w1,w2):
    index=sentence.find(w1)
    word=w1
    if index==-1:
        index=sentence.find(w2)
        word=w2
    if index==-1:
        index,word=None,None
    return index,word

def find_two_generations(w1_tmp,w2_tmp,g):
    start=g.rfind('[')
    end=g.rfind(']')
    if start==-1 or end==-1:
        return None,None,None,None
    g=g[start+1:end]
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
            if len(g1)<3 or len(g2)<3:
                continue
            else:
                break

            
    if len(g1)==0 or len(g2)==0:
        return None,None,None,None
    
    ig1,w1=find_index(g1,w1_tmp,w2_tmp)
    ig2,w2=find_index(g2,w1_tmp,w2_tmp)
    
    if ig1 is None or ig2 is None or w1==w2:
        return None,None,None,None
  
    #g1 and g2 must start and finish by capital letter and period
    #also checking that w1 and w2 are correctly placed.
    w1,w2,g1,g2=check_capital_and_punc(w1,w2,g1,g2,ig1,ig2,use_split=False)
    return g1,ig1,g2,ig2


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,help='path to sentence pair generations for visual tasks',default='visual_task/visual_sentence_generations.txt')
    parser.add_argument("--output_file",type=str,help='path to sentence pairs with filtering visual filtered file',default='visual_task/visual_sentence_generations_with_filtering_prompts')
    return parser.parse_args(argv)

if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    input_file=args.input_file
    output_file=args.output_file
    out=[]
    vocabulary={}
    with open(input_file) as buf:
        lines=buf.readlines()
    seen_words=set() #for some reason some base words are duplicated
    for line in tqdm.tqdm(lines):
        line=line.rstrip().split('|')
        #previous
        bin,w1,w2,_,rule,generation=line
        assert rule=='VISUAL',rule
        #current
        #bin,word,pos,inflection,pos_infl,rule,generation=line
        #formatting the generated sentence from the llm
        g1,ig1,g2,ig2=find_two_generations(w1,w2,generation)
        if g1 is None:
            #findind the two sentences did not work
            continue
        #creating new sentence with a new word inside
        gg1=g1[:ig1]+w2+g1[ig1+len(w1):]
        gg2=g2[:ig2]+w1+g2[ig2+len(w2):]
        assert gg1[ig1:ig1+len(w2)]==w2,(gg1,ig1,w1)
        assert gg2[ig2:ig2+len(w1)]==w1,(gg2,ig2,w2)
        #checking that the word is not in the modified sentences
        #in some cases the target wod is present twice in the original or generated sentences
        if w1 in gg1 or w2 in gg2:
            continue 
       
        #making prompt asking the LLM to solve the quadruplet
        prompt1=make_prompt(g1,gg1)
        answer1='A'
        prompt11=make_prompt(gg1,g1)
        answer11='B'

        prompt2=make_prompt(g2,gg2)
        answer2='A'
        prompt22=make_prompt(gg2,g2)
        answer22='B'
        prompts_list='/'.join((prompt1,prompt11,prompt2,prompt22,answer1,answer11,answer2,answer22))
        out.append('|'.join((str(bin),rule,w1,g1,str(ig1),w2,g2,str(ig2),prompts_list)))
        


    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')