import torch
import numpy as np
import random, json
import torch.nn.functional as F
import tqdm
import os, ast
from utils import pretty_print, get_probs, read_pretraining_data, model_init, get_context
os.environ["TOKENIZERS_PARALLELISM"]='false'
import nltk, sys

if __name__ == '__main__':
    pairs_file=sys.argv[1]
    model_name=sys.argv[2]
    if torch.cuda.is_available():
        cuda=True
    else:
        cuda=False
    use_context=False
    verbose=False 
    if '_100M' in pairs_file:
        pretraining_file='BabyLM_2024_formatted/longtail_100M'
        dictionnary_file='BabyLM_2024_formatted/dictionnary_100M'
        output_dir=os.path.join(os.path.dirname(pairs_file),'results_100M_padtoken')
    elif '_10M' in pairs_file:
        pretraining_file='BabyLM_2024_formatted/longtail_10M'
        dictionnary_file='BabyLM_2024_formatted/dictionnary_10M'
        output_dir=os.path.join(os.path.dirname(pairs_file),'results_10M_padtoken')
    else:
        assert False

    
    if verbose:
        print('using context:',use_context)
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    if True:
        dictionnary={}
        with open(dictionnary_file) as buf:
            lines=buf.readlines()
            for line in lines:
                word,freq=line.rstrip().split(' ')
                assert word not in dictionnary
                dictionnary[word]=int(freq)
    
    if use_context:
        print('reading pretraining file...')
        context_data=read_pretraining_data(pretraining_file)
        print('done')
    pairs={}
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    success,all_pairs={},{}
    for bin in range(len(freq_bins)):
        success[bin]=0
        all_pairs[bin]=0
    with open(pairs_file) as buf:
        pairs=buf.readlines()
    print('number of pairs:',len(pairs))
    inputs,contexts,bins=[],[],[]
    c=0
    prop_context=0
    print('finding context')
    for p in tqdm.tqdm(range(len(pairs))):  
        pair=pairs[p].rstrip().split('|')
        if len(pair)==3:
            sentence_good,sentence_bad,bin=pair
            bin=0
        else:
            sentence_good,sentence_bad,bin,dataset=pair[:4]
            #if dataset=='reflexive':
            #    continue
        bin=int(bin) 

        if use_context:
            tmp_good=sentence_good
            tmp_bad=sentence_bad
            if False:
                #using a random context
                pp=random.randint(0,len(pairs)-1)
                tmp_good,tmp_bad=pairs[pp].split('|')[:2]
            #context_good,context_bad,bin=get_context_pair(context_data,tmp_good,tmp_bad)    
            context_good,_,_=get_context(context_data,tmp_good)
            context_bad,_,_=get_context(context_data,tmp_bad)
            
            if context_good!='' or context_bad!='':
                prop_context+=1    
            sentence_good=' '.join((context_good,sentence_good))
            sentence_bad=' '.join((context_bad,sentence_bad))
        else:    
            #this boost performances more than adding a BOS token
            context_good=' '
            context_bad=' '
            sentence_good=context_good+sentence_good
            sentence_bad=context_bad+sentence_bad

        contexts+=[context_good,context_bad]
        inputs+=[sentence_good,sentence_bad]   
        bins.append(bin)
    
    
    model, tokenizer, loss_fn, bert = model_init(model_name, cuda)  
    if verbose:
        print("Model init",model_name,"with vocab size:",tokenizer.vocab_size)
    else:
        print(model_name)

    batch_size=200*2 #needs to be a multiple of 4 s
    assert len(inputs)%2==0
    assert len(inputs)==len(bins)*2
    assert batch_size%2==0
    nb_batches=int(len(inputs)/batch_size)+1
    print('batches:',nb_batches)
    for i in tqdm.tqdm(range(nb_batches)):
        batch=inputs[i*batch_size:(i+1)*batch_size]
        if len(batch)==0:
            break
        batch_context=contexts[i*batch_size:(i+1)*batch_size]
        bin_batch=bins[i*int(batch_size/2):(i+1)*int(batch_size/2)]
        batch_log_probs=get_probs(model,tokenizer,batch,loss_fn,batch_context,cuda,bert)
        
        batch_log_probs=batch_log_probs.reshape(-1,2)
        for j in range(len(batch_log_probs)):
            prob_g1,prob_g2=batch_log_probs[j]
            bin=bin_batch[j]
            if prob_g1>prob_g2:
                success[bin]+=1
            all_pairs[bin]+=1  
        if verbose and c%20==0:
            pretty_print(success,all_pairs,verbose)
        c+=1
    cout=pretty_print(success,all_pairs,verbose)        
    
    if verbose:
        print('Model:',model_name)
        print('pairs file:',pairs_file)
        print('using context:',use_context)
        print('number of context found:',prop_context/len(pairs))
    else:
        base_model_name=model_name.split('/')[-1]
        cout=[base_model_name]+cout
        with open(os.path.join(output_dir,base_model_name),'w') as buf:
            buf.write(' '.join(cout)+'\n')