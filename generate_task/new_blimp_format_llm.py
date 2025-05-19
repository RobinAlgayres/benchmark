import os,sys
import random
import numpy as np
from format_text import space_characters
from eval_pairs import read_pretraining_data

def removing_word_in_between(split_g1,split_g2,ig1,ig2):
    det1,det2=split_g1[ig1-1],split_g2[ig2-1]
    if det1.lower()!='the':
        if ig1>=2 and split_g1[ig1-2].lower()=='the':
            split_g1=split_g1[:ig1-2]+split_g1[ig1-1:]
            g1=' '.join(split_g1)
            ig1=ig1-1
        else:
            #print(pos,w1,w2,'G1',g1,'G2',g2)
            return None,None,None,None
    else:
        g1=' '.join(split_g1)
    if det2.lower()!='the':
        if ig2>=2 and split_g2[ig2-2].lower()=='the':
            split_g2=split_g2[:ig2-2]+split_g2[ig2-1:]
            g2=' '.join(split_g2)
            ig2=ig2-1
        else:
            return None,None,None,None    
    else:
        g2=' '.join(split_g2)
    return g1,g2,ig1,ig2

def find_reflexive(split_g1,split_g2):
    pronouns=['himself','itself','herself','themselves']
    r1,r2=-1,-1
    for pronoun in pronouns:
        if pronoun in split_g1:
            r1=split_g1.index(pronoun)
        if pronoun in split_g2:
            r2=split_g2.index(pronoun)
    if r1==-1 or r2==-1:
        return None,None
    split_g1=split_g1[:r1+1]
    split_g2=split_g2[:r2+1]
    if w1 not in split_g1 or w2 not in split_g2:
        return None,None
    return split_g1,split_g2

def find_verb(g1,g2,ig1,ig2,w1,w2):
    
    split_g1=g1.split(' ')
    split_g2=g2.split(' ')
    if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
        singular=split_g2
        plural=split_g1
    elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
        singular=split_g1
        plural=split_g2
    else:
        assert False,(w1,w2)

    #finding i_s and i_p that are the index of the verbs
    i_s=-1
    vs,vp=None,None
    bes_dict={'are':'is','have':'has','do':'does','dont':'doesnt','arent':'isnt'}
    neg_dict={'are':'isnt','have':'hasnt','do':'doesnt','dont':'does','arent':'is'}
    for i_p in range(len(plural)):
        w=plural[i_p]
        if w in [w1,w2]:
            continue
        elif w in bes_dict:
            vp=w
            vs=bes_dict[w]
            if vs in singular:
                i_s=singular.index(vs)
            elif neg_dict[w] in singular:
                vs=neg_dict[w]
                i_s=singular.index(vs)
            else:
                #print(singular,plural,vs,vp)
                return None,None,None,None
            break
        elif w+'s' in singular:
            vp=w
            vs=w+'s'
            i_s=singular.index(vs)
            break
        elif w+'es' in singular:
            vp=w
            vs=w+'es'
            i_s=singular.index(vs)
            break
        elif w[-1]=='y' and w[:-1]+'ies' in singular:
            vp=w
            vs=w[:-1]+'ies'
            i_s=singular.index(vs)
            break

    if i_s==-1:
        #print(singular,plural,i_s,i_p)
        return None,None,None,None
    singular=singular[:i_s+1]
    plural=plural[:i_p+1]
    gs=' '.join(singular)    
    gp=' '.join(plural)
    if w1 in singular and w2 in plural:
        g1,g2=gs,gp
        v1,v2=vs,vp
    elif w1 in plural and w2 in singular:
        g1,g2=gp,gs
        v1,v2=vp,vs
    else:
        return None,None,None,None
    return v1,v2,g1,g2

def format_sentence(sentence,word,dictionnary):
    sentence=list(filter(None, sentence.split(' ')))
    formatted_generation=[]
    unknown_word=False
    for genword in sentence:
        genword,_,_,_=space_characters(genword,map_letters)
        if genword==word:
            formatted_generation+=[genword] 
            continue
        elif genword is None:
            unknown_word=True
            break
        genword=genword.split(' ')
        for sub_genword in genword:
            if sub_genword.lower() not in dictionnary:
                unknown_word=True
                break
        formatted_generation+=genword 
    
    #print(formatted_generation)
    if word not in formatted_generation:
       # print(word,formatted_generation,sentence)
        return ' '.join(sentence),None,True
    index=formatted_generation.index(word)
    formatted_generation=' '.join(formatted_generation)
    return formatted_generation, index , unknown_word

if __name__=='__main__':
    p=0
    map_letters={}
    map_letters['foreigns']=[]
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['symbols']='!"$%&\'()*,-.0123456789:;?@[]'
    map_letters['all']=map_letters['letters']+map_letters['symbols']


    corpus='100M'
    frequency_file='BabyLM_2024_formatted/minimal_ltblimp_'+corpus+'_generations.txt'
    word_pair_file='BabyLM_2024_formatted/minimal_ltblimp_'+corpus+'_task'
    pretraining_file='BabyLM_2024_formatted/longtail_'+corpus
    dictionnary_file='BabyLM_2024_formatted/dictionnary_'+corpus
    dictionnary=set()
    #context_data=None
    context_data=read_pretraining_data(pretraining_file)
    if os.path.isfile(word_pair_file):
        #assert False,(word_pair_file,'already exist')
        pass


    with open(dictionnary_file) as buf:
        lines=buf.readlines()
        for line in lines:
            dictionnary.add(line.rstrip().lower())
    c=0
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    output,output2=[],[]
    with open(frequency_file) as buf:
        lines=buf.readlines()

   

    for line in lines:
        line=line.rstrip().split('|')

        if len(line)==6:
            bin,w1,w2,pos,_,g=line
            if g=='[empty]':
                continue
            if len(g)<3:
                continue
            if g[0]!='[' or g[-1]!=']':
                continue
            if g.count("[") != 1 or g.count("]") != 1:
                continue
            
            g=g[1:-1]
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
                    continue
            g2,ig2,unknown_w2=format_sentence(g2,w2,dictionnary)
            if unknown_w2:
                continue

        else:
            #g1,ig1,unknown_w1=format_sentence(g1,w1,dictionnary)
            #g2,ig2,unknown_w2=format_sentence(g2,w2,dictionnary)
            #if unknown_w1 or unknown_w2:
            continue

        if len(g1.split(' '))<3 or len(g2.split(' '))<3:
            continue
        split_g1=g1.split(' ')
        assert split_g1[ig1]==w1
        split_g2=g2.split(' ')
        assert split_g2[ig2]==w2
        v1,v2=None,None
        if 'ANAPHORA' in pos:
            #make sure that there is no third person verb in between the noun and reflexive pronoun
            if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
                pass
            elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
                pass
            else:
                print('not singular,plural',pos,w1,w2)
                continue
            _,_,gg1,gg2=find_verb(g1,g2,ig1,ig2,w1,w2)
            if gg1 is not None:
                #print(gg1,gg2,w1,w2)
                p+=1
                continue
            split_g1,split_g2=find_reflexive(split_g1,split_g2)
            if split_g1 is None:
                continue
            
            if 'is' in split_g1 or 'are' in split_g1 or 'has' in split_g1 or 'have' in split_g1 or 'was' in split_g1 or 'were' in split_g1:    
                continue
            if 'is' in split_g2 or 'are' in split_g2 or 'has' in split_g2 or 'have' in split_g2 or 'was' in split_g2 or 'were' in split_g2:
                continue
            if split_g1[0].lower() in ['this','those','these'] or split_g2[0].lower() in ['this','those','these']:
                continue
            g1=' '.join(split_g1)
            g2=' '.join(split_g2)

        elif 'DET' in pos:
            if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
                pass
            elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
                pass
            else:
                print('not singular,plural',pos,w1,w2)
                continue
            if ig2==0 or ig1==0:
                continue
            if split_g1[ig1-1].lower() not in ['that','these','this','those']:
                continue
            if split_g2[ig2-1].lower() not in ['that','these','this','those']:
                continue
            split_g1=split_g1[ig1-1:ig1+1]
            split_g2=split_g2[ig2-1:ig2+1]
            g1=' '.join(split_g1)
            g2=' '.join(split_g2)
            ig1=1
            ig2=1
            if 'itself' in split_g1 or 'herself' in split_g1 or 'himself' in split_g1 or 'themselves' in split_g1:
                continue
            if 'itself' in split_g2 or 'herself' in split_g2 or 'himself' in split_g2 or 'themselves' in split_g2:
                continue

        elif 'SV' in pos:
            #we ll have to make sure there are no reflexive pronouns 
            #and that there is a verb in third person / first person after the noun
            if 'VERB' in pos:
                #removing verb based SV agreements, not easy to control
                continue
            if (w1[-1]=='s' and w2[-1]!='s') or (w1[-2:]=='es' and w2[-1]=='s'):
                pass
            elif (w2[-1]=='s' and w1[-1]!='s') or (w2[-2:]=='es' and w1[-1]=='s'):
                pass
            else:
                print('not singular,plural',pos,w1,w2)
                continue
            if 'itself' in split_g1 or 'herself' in split_g1 or 'himself' in split_g1 or 'themselves' in split_g1:
                continue
            if 'itself' in split_g2 or 'herself' in split_g2 or 'himself' in split_g2 or 'themselves' in split_g2:
                continue
            if split_g1[0].lower() in ['this','those','these']:
                #print(pos,g1)
                continue
            if split_g2[0].lower() in ['this','those','these']:
                #print(pos,g2)
                continue
            v1,v2,g1,g2=find_verb(g1,g2,ig1,ig2,w1,w2)
            if g1 is None:
                continue
        if 'LONG' in pos:
            if 'that can be' not in g1 or 'that can be' not in g2:
                print(pos,g1,g2)
                continue

        if 'SV' in pos or 'ANA' in pos or 'DET' in pos:
            #we do not want other plural marks
            for pattern in ['multiple', 'several','many']:
                if pattern in g1:
                    g1=g1.split(' ')
                    ind=g1.index(pattern)
                    g1=g1[:ind]+g1[ind+1:]   
                    g1=' '.join(g1) 
                    if ind<ig1:
                        ig1=ig1-1
                if pattern in g2:
                    g2=g2.split(' ')
                    ind=g2.index(pattern)
                    g2=g2[:ind]+g2[ind+1:]
                    g2=' '.join(g2) 
                    if ind<ig2:
                        ig2=ig2-1

        s1,s2,i1,i2='s1','s2','0','0'
        #g1 and g2 must start and finish by capital letter and period
        #also checking that w1 and w2 are correctly placed.
        w1,w2,g1,g2=check_capital_and_punc(w1,w2,g1,g2,ig1,ig2)
        if g1 is None:
            continue
        #finding frequency bins
        if context_data:
            f1=context_data[w1]['freq']
            f2=context_data[w2]['freq']
            b1=np.where(f1>=freq_bins)[0][-1]
            b2=np.where(f2>=freq_bins)[0][-1]
            assert b1!=0 or b2!=0,(f1,f2,w1,w2) 
            if v1 is not None:
                f3,b3=context_data.get(v1,None),10
                f4,b4=context_data.get(v2,None),10
                if f3 is not None:
                    b3=np.where(f3['freq']>=freq_bins)[0][-1]
                if f4 is not None:
                    b4=np.where(f4['freq']>=freq_bins)[0][-1]
                bin=str(min(b1,b2,b3,b4))
            else:
                bin=str(min(b1,b2))
        else:
            bin=str(0)

        output.append('|'.join((bin,pos,w1,s1,i1,g1,str(ig1),w2,s2,i2,g2,str(ig2))))
        c+=1
    print('total correct sentences',c,len(lines))
    print(p)
    print(word_pair_file)
    with open(word_pair_file,'w') as buf:
        buf.write('\n'.join(output)+'\n')

        