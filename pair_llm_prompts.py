import os,sys
import random

def make_prompt(s,gA,gB):
    prompt=("I have invented one new english word \'blick\' that you can use as in the following sentence:_"
        "<start of sentence> "+s+"<end of sentence>_"
        "Now I give you two new sentences A and B:_"
        "<start of sentence A> "+gA+"<end of sentence A>_"
        "<start of sentence B> "+gB+"<end of sentence B>_"
        "Which of the sentence A or B uses the word 'blick' correctly? Put your answer, A or B, in between brackets.")
    return prompt

def add_blick(s,i,w):
    return s[:i]+'blick'+s[i+len(w):]

#llm prompt to create alternative sentences.
pair_file=sys.argv[1]#'BabyLM_2024_formatted/pairs_100M'
prompt_file=sys.argv[2]#'BabyLM_2024_formatted/pairs_100M_prompts'


with open(pair_file) as buf:
    lines=buf.readlines()
pairs={}
prompts=[]
for line in lines:
    bin,pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2=line.rstrip().split('|')
    i1,ig1,i2,ig2=int(i1),int(ig1),int(i2),int(ig2)
    assert s1[i1:i1+len(w1)]==w1,(w1,s1[i1:i1+len(w1)])
    assert s2[i2:i2+len(w2)]==w2,(w2,s2[i2:i2+len(w2)])
    assert g1[ig1:ig1+len(w1)]==w1,(w1,g1[ig1:ig1+len(w1)])
    assert g2[ig2:ig2+len(w2)]==w2,(w2,g2[ig2:ig2+len(w2)])
    
    ss1=add_blick(s1,i1,w1)
    ss2=add_blick(s2,i2,w2)
    gg1=add_blick(g1,ig1,w1)
    gg2=add_blick(g2,ig2,w2)
    
    
    #creating new sentence with a new word inside
    #ss1,ss2,gg1,gg2=s1.split(' '),s2.split(' '),g1.split(' '),g2.split(' ')
    
    #assert ss1[i1]==w1 and gg1[ig1]==w1
    #assert ss2[i2]==w2 and gg2[ig2]==w2

    #changing w1 and w2 into blick
    #ss1[i1],gg1[ig1],ss2[i2],gg2[ig2]='blick','blick','blick','blick'
    #ss1,gg1,ss2,gg2=' '.join(ss1),' '.join(gg1),' '.join(ss2),' '.join(gg2)
    
    #checking that the word is not in the modified sentences
    #in some cases the target wod is present twice in the original or generated sentences
    if w1 in ss1 or w1 in gg1:
        continue
    if w2 in ss2 or w2 in gg2:
        continue

    prompt1=make_prompt(ss1,gg1,gg2)
    answer1='A'
    prompt11=make_prompt(ss1,gg2,gg1)
    answer11='B'

    prompt2=make_prompt(ss2,gg2,gg1)
    answer2='A'
    prompt22=make_prompt(ss2,gg1,gg2)
    answer22='B'
    
    
    prompts.append('|'.join((bin,pos,w1,s1,str(i1),g1,str(ig1),w2,s2,str(i2),g2,str(ig2),prompt1,prompt11,prompt2,prompt22,answer1,answer11,answer2,answer22)))

print(prompt_file,len(prompts))
with open(os.path.join(prompt_file),'w') as buf:
    buf.write('\n'.join(prompts)+'\n')




    

    
