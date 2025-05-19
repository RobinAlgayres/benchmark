import numpy as np
import tqdm
import random
import os
import torch
from utils import model_init, get_probs, get_context, read_pretraining_data
from utils import check_capital_and_punc, pretty_print, pretty_print_avg
os.environ["TOKENIZERS_PARALLELISM"]='false'
import sys,json

def use_madeup_inflswap(w1,w2,g1,g2,pos):
    oldw1,oldw2=w1,w2
    madeup='tartouf'
    if w1[-2:]=='ed' and pos=='VERB':
        w1=madeup+'ed'
    elif w1[-1:]=='s'and pos!='ADJ':
        w1=madeup+'s'
    elif w1[-3:]=='ing' and pos=='VERB':
        w1=madeup+'ing' 
    else:
        w1=madeup

    if w2[-2:]=='ed' and pos=='VERB':
        w2=madeup+'ed'
    elif w2[-1:]=='s' and pos!='ADJ':
        w2=madeup+'s'
    elif w2[-3:]=='ing' and pos=='VERB':
        w2=madeup+'ing' 
    else:
        w2=madeup
    #if oldw1[:2]=='un':
    #    w1='un'+w1
    #elif oldw2[:2]=='un':
    #    w2='un'+w2
    #formatting did not work
    if w1==w2:
        return None,None,None,None,None,None
    #w1,w2=oldw1,oldw2
    #changing w1 and w2 in generated sentences by their new form    
    g1,g2=g1.split(' '),g2.split(' ')
    g1[ig1],g2[ig2]=w1,w2
    g1,g2=' '.join(g1),' '.join(g2) 
    return w1,w2,g1,g2,oldw1,oldw2

def use_madeup_wordswap(w1,w2,g1,g2,s1,s2):
      
    g1,g2=g1.split(' '),g2.split(' ')
    s1,s2=s1.split(' '),s2.split(' ')
    iw1=s1.index(w1)
    iw2=s2.index(w2)

    #w1,w2='blick','dax'
    w1,w2='dax','blick'
    g1[ig1],g2[ig2]=w1,w2
    s1[iw1],s2[iw2]=w1,w2
    g1,g2=' '.join(g1),' '.join(g2) 
    s1,s2=' '.join(s1),' '.join(s2) 
    return w1,w2,g1,g2,s1,s2



def inference(inputs,model,tokenizer,loss_fn,verbose=True,bert=None):
    div=4
    batch_size=100*div
    assert len(inputs)%div==0
    assert len(inputs)==len(bins)*div
    assert batch_size%div==0
    save_pairs=False

    for i in tqdm.tqdm(range(int(len(inputs)/batch_size)+1)):
        batch=inputs[i*batch_size:(i+1)*batch_size]
        if len(batch)==0:
            continue
        batch_context=contexts[i*batch_size:(i+1)*batch_size]
        batch_pairs=tmp_pairs[i*int(batch_size/div):(i+1)*int(batch_size/div)]
        bin_batch=bins[i*int(batch_size/div):(i+1)*int(batch_size/div)]
        batch_log_probs=get_probs(model,tokenizer,batch,loss_fn,batch_context,cuda,bert)
        batch_log_probs=batch_log_probs.reshape(-1,div)
        
        for j in range(len(batch_log_probs)):
            prob_g1,prob_g2,prob_b1,prob_b2=batch_log_probs[j]
            bin,pos=bin_batch[j][:2]
            if save_pairs:
                if prob_g1>prob_b1 and prob_g2>prob_b2:
                    selected_pairs.append(batch_pairs[j])
                
            if prob_g1>prob_b1:# and prob_g2>prob_b2:
                success[bin]+=1
                pos_success[bin][pos]+=1
                
                   
            if prob_g2>prob_b2:
                success[bin]+=1
                pos_success[bin][pos]+=1
            all_pairs[bin]+=2
            pos_all_pairs[bin][pos]+=2
        if verbose and (i-5)%15==0:
            pretty_print_avg(pos_success,pos_all_pairs,verbose) 
           
            
    cout=pretty_print_avg(pos_success,pos_all_pairs,verbose) 

    if save_pairs:
        print('writing filtered final')
        with open('final10M_syntax','w') as buf:
            buf.write('\n'.join(selected_pairs)+'\n')
    return cout
    
def swap_words(w1,ig1,g1,w2,ig2,g2):
    try:
        assert g1[ig1:ig1+len(w1)]==w1,(w1,g1[ig1:ig1+len(w1)])
        assert g2[ig2:ig2+len(w2)]==w2,(w2,g2[ig2:ig2+len(w2)])
        gg1=g1[:ig1]+w2+g1[ig1+len(w1):]
        gg2=g2[:ig2]+w1+g2[ig2+len(w2):]
    except:
        gg1,gg2=g1.split(' '),g2.split(' ')
        assert gg1[ig1].lower()==w1,(w1,g1)
        assert gg2[ig2].lower()==w2,(w2,g2)
        gg1[ig1],gg2[ig2]=w2,w1
        gg1,gg2=' '.join(gg1),' '.join(gg2) 
    
    #if the swapped word is the first one, the sentence may not start with upper case anymore
    gg1=gg1[0].upper()+gg1[1:]
    gg2=gg2[0].upper()+gg2[1:]
    return gg1,gg2

if __name__ == '__main__':
    pairs_file=sys.argv[1]
    model_name=sys.argv[2]
    if torch.cuda.is_available():
        cuda=True
    else:
        cuda=False
    use_context=False
    use_all_pos=False
    verbose=True
    if '_100M' in pairs_file:
        pretraining_file='BabyLM_2024_formatted/longtail_100M'
        dictionnary_file='BabyLM_2024_formatted/dictionnary_100M'
        output_dir='results_100M_matrix'#_use_context'
        '''
        model_name_list=[
                         'thu-coai/CDial-GPT_LCCC-large',
                         'babylm/babyllama-100m-2024',
                         'babylm/t5-base-strict-2023',
                         'babylm/opt-125m-strict-2023',
                         'phonemetransformers/GPT2-85M-BPE-TXT',
                         'phonemetransformers/GPT2-85M-CHAR-TXT',
                         'bbunzeck/grapheme-llama',
                         'bbunzeck/baby_llama',
                         'gpt2',
                         '/datasets/pretrained-llms/Llama-3.2-1B/',
                         'babylm/ltgbert-100m-2024',
                         'ltg/gpt-bert-babylm-base',
                         'babylm/roberta-base-strict-2023',
                         'SzegedAI/babylm-strict-mlsm', #deberta 
                         'SzegedAI/babylm24_LSM_strict', #deberta
                         'SrikrishnaIyer/RoBERTa_WML_distill-Babylm-100M-2024', #roberta
                        ]
        '''
    elif '_10M' in pairs_file:
        pretraining_file='BabyLM_2024_formatted/longtail_10M'
        dictionnary_file='BabyLM_2024_formatted/dictionnary_10M'
        output_dir='results_10M_matrix'#_use_context'
        '''
        model_name_list=[
                         'babylm/babyllama-10m-2024',
                         'JLTastet/baby-llama-2-345m',
                         'babylm/t5-base-strict-small-2023',
                         'babylm/roberta-base-strict-small-2023',
                         'babylm/opt-125m-strict-small-2023', 
                         'gpt2',
                         '/datasets/pretrained-llms/Llama-3.2-1B/',
                         'SzegedAI/babylm-strict-small-mlsm',
                         'SzegedAI/babylm24_LSM_strict-small',
                         'SrikrishnaIyer/RoBERTa_WML_distill-Babylm-10M-2024',
                         'babylm/ltgbert-10m-2024',
                         'ltg/gpt-bert-babylm-base',
                        ]
        '''
    else:
        assert False

    if 'inflswap' in pairs_file:
        output_dir=os.path.join(output_dir,'inflswap')
    elif 'syntax' in pairs_file:
        output_dir=os.path.join(output_dir,'syntax')
    elif 'wordswap' in pairs_file:
        output_dir=os.path.join(output_dir,'wordswap')
    elif 'feasable' in pairs_file:
        output_dir=os.path.join(output_dir,'feasable')
    else:
        assert False,pairs_file

    if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
    if verbose:
        print('using context:',use_context)
    

    dictionnary={}
    with open(dictionnary_file) as buf:
        lines=buf.readlines()
        for line in lines:
            word,freq=line.rstrip().split(' ')
            assert word not in dictionnary
            dictionnary[word]=int(freq)

    if use_context and False:
        if verbose:
            print('reading pretraining file...')
        context_data=read_pretraining_data(pretraining_file)
    pairs={}
    with open(pairs_file) as buf:
        pairs=buf.readlines()
    success,pos_success,all_pairs,pos_all_pairs={},{},{},{}
    inputs,contexts,bins=[],[],[]
    c=0
    prop_context=0
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    for bin in range(len(freq_bins)-1):
        success[bin]=0
        all_pairs[bin]=0
        pos_success[bin]={}
        pos_all_pairs[bin]={}
        if not use_all_pos:
            for pos_tag in ['ADJ','VERB','NOUN','LONG','SHORT','ANAPHORALONG','SVLONG','ANAPHORASHORT','SVSHORT','DET','SV','ANAP','SYNTAX']:
                pos_success[bin][pos_tag]=0
                pos_all_pairs[bin][pos_tag]=0
    selected_pairs,tmp_pairs=[],[]

    for p in tqdm.tqdm(range(len(pairs))):    
        pair=pairs[p].rstrip()   
        bin,pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2=pair.split('|')
        ig1,ig2=int(ig1),int(ig2)
        
        #if p%2==0:
        #    continue
        # make words in lower case
        # make sentences start with upper case and finish with period
        # also asserting the words are correctly placed in the sentences 
        #w1,w2,g1,g2=check_capital_and_punc(w1,w2,g1,g2,ig1,ig2)
  
        
        if True:
            f1,f2=dictionnary.get(w1,0),dictionnary.get(w2,0)
            b1=np.where(f1>=freq_bins)[0][-1]
            b2=np.where(f2>=freq_bins)[0][-1]
            #if abs(b2-b1)>1:
            #this destroy the frequency effect  
            #    continue
            bin=int(round(np.min([b1,b2])))
        bin=int(bin)
        if bin>1:
            continue
        assert w1!=w2,pair

        if not use_all_pos:
            pos=pos.split('_')[0].split('-')[0]
            if 'inflswap' in pairs_file or 'wordswap' in pairs_file:
                if pos=='ADJ':
                    #not enough ADJ per bin
                    continue
            if 'syntax' in pairs_file:
                if 'DET' in pos:
                    pos='SHORT'
                elif 'SHORT' in pos:
                    pos='SHORT'
                elif 'LONG' in pos:
                    pos='LONG'
                else:
                    continue
        else:
            if pos not in pos_success[bin]:
                pos_success[bin][pos]=0
                pos_all_pairs[bin][pos]=0
        
        
        

        if False:
            #if 'VERB' not in pos:
            #    continue
            #w1,w2,g1,g2,_,_=use_madeup_inflswap(w1,w2,g1,g2,pos)
            w1,w2,g1,g2,s1,s2=use_madeup_wordswap(w1,w2,g1,g2,s1,s2)
            if w1 is None: 
                continue

        gg1,gg2=swap_words(w1,ig1,g1,w2,ig2,g2)    
       
        sentence_good_1,sentence_good_2=g1,g2
        sentence_bad_1,sentence_bad_2=gg1,gg2

        if use_context:
            if True:
                #using the true context
                context_good_1,context_good_2=s1,s2
                context_bad_1,context_bad_2=s2,s1
            else:
                tmp_good_1,tmp_good_2=sentence_good_1,sentence_good_2
                tmp_bad_1,tmp_bad_2=sentence_bad_1,sentence_bad_2
                if False:
                    #sampling random context
                    pp=random.randint(0,len(pairs)-1)
                    _,_,_,tmp_good_1,_,_,_,_,tmp_bad_1,_,_,_=pairs[pp].split('|')
                    pp=random.randint(0,len(pairs)-1)
                    _,_,_,tmp_good_2,_,_,_,_,tmp_bad_2,_,_,_=pairs[pp].split('|') 
                context_good_1,_,_=get_context(context_data,tmp_good_1)
                context_good_2,_,_=get_context(context_data,tmp_good_2)
                context_bad_1,_,_=get_context(context_data,tmp_bad_1)
                context_bad_2,_,_=get_context(context_data,tmp_bad_2)
            
            #context_good_1,context_good_2,context_bad_1,context_bad_2='','','',''
            if context_good_1!='' or context_good_2!='':
                prop_context+=1

            sentence_good_1=' '.join((context_good_1,sentence_good_1))
            sentence_bad_1=' '.join((context_bad_1,sentence_bad_1))
            sentence_good_2=' '.join((context_good_2,sentence_good_2))
            sentence_bad_2=' '.join((context_bad_2,sentence_bad_2))
        else:
            context_good_1,context_good_2=' ',' '#' .',' .'
            context_bad_1,context_bad_2=' ',' '#' .',' .'

            sentence_good_1=context_good_1+sentence_good_1
            sentence_bad_1=context_bad_1+sentence_bad_1
            sentence_good_2=context_good_2+sentence_good_2
            sentence_bad_2=context_bad_2+sentence_bad_2

        contexts+=[context_good_1,context_good_2,context_bad_1,context_bad_2]
        inputs+=[sentence_good_1,sentence_good_2,sentence_bad_1,sentence_bad_2]  
        bins.append((bin,pos,w1,w2,sentence_good_1,sentence_bad_1,sentence_good_2,sentence_bad_2))
        tmp_pairs.append(pair)
        #if len(inputs)>100:
        #    break
        
    if verbose:
        print('number of context found:',prop_context,'out of',len(inputs)/4,len(pairs),len(tmp_pairs))
    
    
    model, tokenizer, loss_fn, bert = model_init(model_name, cuda)  
    if verbose:
        print("Model init",model_name,"with vocab size:",tokenizer.vocab_size)
    else:
        print(model_name)
        
    cout=inference(inputs,model,tokenizer,loss_fn,verbose,bert)
    
    if verbose:
        print('Model:',model_name)
        print('pairs file:',pairs_file)
        print('using context:',use_context)
        print('number of context found:',prop_context/len(pairs))
    else:
        if 'Llama-3' in model_name:
            base_model_name=model_name.split('/')[-2]
        else:
            base_model_name=model_name.split('/')[-1]
        cout['MODEL']=base_model_name
        with open(os.path.join(output_dir,base_model_name+".json"),'w') as buf:
            json.dump(cout,buf)
