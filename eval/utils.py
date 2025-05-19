import torch, tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModel, Qwen2ForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel,T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer
from get_hist import format_pos
from format_text import space_characters
import numpy as np
import ast, nltk

def model_init(model_string, cuda):
    bert=None
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    if 'gpt-bert' in model_string:
        bert='GPT-BERT'
    elif 'ltg' in model_string:
        bert='BERT'
    elif 'roberta' in model_string:
        bert='BERT'
    elif 'SzegedAI' in model_string:
        bert='BERT'
    elif 'SrikrishnaIyer' in model_string:
        bert='BERT'
    elif 'antlm' in model_string:
        bert='BERT'
    elif 't5-' in model_string:
        bert=None


    if 'Llama-3' in model_string:
        tokenizer = AutoTokenizer.from_pretrained(model_string,local_files_only=True)
        model = LlamaForCausalLM.from_pretrained(model_string,local_files_only=True,device_map = 'auto')
    elif 'QwQ' in model_string:
        tokenizer = AutoTokenizer.from_pretrained(model_string,local_files_only=True)
        model = Qwen2ForCausalLM.from_pretrained(model_string,local_files_only=True,device_map = 'auto')
    elif 'antlm' in model_string:
        tokenizer = AutoTokenizer.from_pretrained(model_string,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_string,trust_remote_code=True)
    elif 't5-' in model_string:
        tokenizer = T5Tokenizer.from_pretrained(model_string)
        model = T5ForConditionalGeneration.from_pretrained(model_string)
    elif 'CDial' in model_string:
        tokenizer = BertTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    elif bert is None:
        tokenizer = AutoTokenizer.from_pretrained(model_string,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_string,trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_string,trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_string,trust_remote_code=True)
    model.eval()
    if cuda: 
        model.to('cuda')
        loss_fn=loss_fn.to('cuda')
    model.config.use_cache=False
    
    return model, tokenizer, loss_fn, bert


def score_bert(model, tensor_input, attention_mask, mask_token_id,loss_fn, bert, device,tokenizer,current_context_mask):
    
    nb_words=torch.sum(attention_mask).int()
    nb_context_words=torch.sum(current_context_mask).int()
    repeat_input = tensor_input.repeat(nb_words-1, 1)
    attention_mask = attention_mask.repeat(nb_words-1, 1)
    #masking the +1 diagonal
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:nb_words-1].to(device)
    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)

    if bert=='BERT':
        #labels for regular BERT        
        labels = repeat_input.masked_fill( masked_input != mask_token_id, -100)
        
    elif bert=='GPT-BERT':
        #labels for GPT-BERT
        repeat_labels = tensor_input.repeat(nb_words-1, 1)
        repeat_labels[:,:-1]=repeat_labels[:,1:]
        repeat_labels[:,-1]=tokenizer.encode(tokenizer.eos_token)[0]
        #getting the real diagonal
        label_mask = torch.ones(tensor_input.size(-1)).diag(0)[:nb_words-1].to(device)
        labels = repeat_labels.masked_fill( label_mask != 1, -100)
    else:
        assert False,bert
    if nb_context_words>2:
        #assert False,(nb_context_words,current_context_mask)
        masked_input=masked_input[nb_context_words:,:]
        attention_mask=attention_mask[nb_context_words:,:]
        labels=labels[nb_context_words:,:]
    with torch.inference_mode():
        outputs = model(masked_input, attention_mask=attention_mask)

    logits=outputs['logits']
    batch_size,_,vocab_size=logits.size()  
    loss=loss_fn(logits.reshape(-1, vocab_size),labels.reshape(-1)).reshape(batch_size,-1)
    #there is only one sentence, we want one loss score
    loss=torch.sum(loss)#/torch.sum(mask)
    
    return loss




def get_loss(inputs,attention_masks,labels,loss_fn,model):
    with torch.no_grad():
        outputs = model(inputs,attention_mask=attention_masks)
    logits=outputs['logits']
    try:
        batch_size,_,vocab_size=logits.size()  
    except:
        batch_size,seq_len=inputs.size()
        logits=logits.reshape(batch_size,seq_len,-1)
        vocab_size=logits.size(-1)  
    loss=loss_fn(logits.reshape(-1, vocab_size),labels.reshape(-1)).reshape(batch_size,-1)
    return loss

def get_probs(model, tokenizer, sentences, loss_fn, contexts, cuda=False, bert=None):
    if cuda:
        device='cuda'
    else:
        device='cpu'
    #tokenizer.pad_token = tokenizer.eos_token
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token ='[PAD]'
    #    tokenizer.eos_token ='[PAD]'
    #print(tokenizer.eos_token,tokenizer.pad_token)
    #l
    inputs=tokenizer(sentences, return_tensors='pt', padding=True)
    inputs['input_ids']=inputs['input_ids'].to(device)
    inputs['attention_mask']=inputs['attention_mask'].to(device)
    labels=inputs['input_ids'].clone()
    batch_size,seq_len=inputs['input_ids'].size()
    
    #padding context mask to the size of the sentence
    context_tokens=tokenizer(contexts, return_tensors='pt', padding=True,add_special_tokens=False)
    contexts_mask=torch.zeros((batch_size,seq_len)).to(device)
    mean_number_context_words=[]    
   
    for i in range(len(context_tokens['input_ids'])):
        assert torch.sum(inputs['attention_mask'][i])>torch.sum(context_tokens['attention_mask'][i])
        nb_context_tokens=torch.sum(context_tokens['attention_mask'][i]).int()+1 #adding one for the BOS/CLS token
        contexts_mask[i,:nb_context_tokens]=1
        mean_number_context_words.append(nb_context_tokens)
    mean_number_context_words=torch.tensor(mean_number_context_words)
    mean_number_context_words=torch.mean(mean_number_context_words.float()).int()
    
    if bert is not None :
        loss=[]
        for i in range(len(sentences)):
            attention_mask=inputs['attention_mask'][i]
            current_context_mask=contexts_mask[i]
            assert torch.sum(attention_mask)>torch.sum(current_context_mask)
            tmp=score_bert(model,inputs['input_ids'][i],attention_mask,tokenizer.mask_token_id,loss_fn,bert,device,tokenizer,current_context_mask)
            loss.append(tmp)
        log_probs=-torch.tensor([loss])
    else:
        
        labels=inputs['input_ids'].clone()
        labels[:,:-1]=inputs['input_ids'][:,1:]
        labels[:,-1]=tokenizer.encode(tokenizer.eos_token)[0]
        loss=get_loss(inputs['input_ids'],inputs['attention_mask'],labels,loss_fn,model)
        #intersection of non-padded BPEs and non-context BPEs
        #because we do not want to compute the loss on the context
        #nor on the padded tokens
        inputs_mask=inputs['attention_mask']
        if mean_number_context_words>2:
            inputs_mask=inputs_mask*(1-contexts_mask)
        #not computing loss on predicting end of sentence, it hads noise
        for i in range(len(loss)):
            index=len(tokenizer.encode(sentences[i]))-1
            inputs_mask[i,index]=0
        #applying mask and computing log probs  
        loss=loss*inputs_mask 
        log_probs=-torch.sum(loss,dim=1)#/denom
        #FOR DEBUG
        #for i in range(len(inputs['input_ids'])):
        #   print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][i]))
        
    assert not torch.isnan(torch.sum(log_probs)),(log_probs,sentences)
    assert not torch.isinf(torch.sum(log_probs)),(log_probs,sentences)
    return log_probs.cpu()
       


def read_pretraining_data(pretraining_file):
    with open(pretraining_file) as buf:
        lines=buf.readlines()
    freqs={}
    for line in tqdm.tqdm(lines):
        assert '|' not in line,line
        data=ast.literal_eval(line)
        freqs[data['word']]=data
        #adding inflections as entries without frequencies
        for inflection in data['all_inflections']:
            if inflection not in freqs:  
                pos=nltk.pos_tag([inflection])[0][1]
                pos=format_pos(inflection,pos)
                freqs[inflection]={'freq':0,'all_inflections':data['all_inflections'],'POS':{pos:{'freq':0}},'context':''}
               
    return freqs

def format_sentence(sentence):
    map_letters={}
    map_letters['foreigns']=[]
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['symbols']='!"$%&\'()*,-.0123456789:;?@[]'
    map_letters['all']=map_letters['letters']+map_letters['symbols']

    new_sentence=[]
    for word in sentence.split(' '):
        word,_,_=space_characters(word,map_letters)
        if word is None:
            continue
        new_sentence.append(word)
    return ' '.join(new_sentence)

def format_context(original_context,word,max_sentence_len=50):
    #get max number of words from sentence so that it surrounds 
    #target word
    context=original_context.lower().split(' ')
    assert word in context
    if len(context)>max_sentence_len:
        index=context.index(word)
        start_ind=max(0,index-int(max_sentence_len/2))
        end_ind=min(len(context),index+int(max_sentence_len/2))
        missing_words=max_sentence_len-(end_ind-start_ind)
        if missing_words>0:
            if start_ind==0:
                end_ind+=missing_words
            else:
                start_ind-=missing_words 
        original_context=original_context.split(' ')[start_ind:end_ind]
        assert len(original_context)>(max_sentence_len/2)
        assert len(original_context)<=max_sentence_len+1
        original_context=' '.join(original_context)
    return original_context

def get_context_util(word,pos,context_data,use_inflections=True):
    pos=format_pos(word,pos)
    if word not in context_data:
        #current word has been seen 0 times in the whole corpus
        #and is not an inflection of any known word in the corpus
        return None,None,0
    #if the word exists with this POS tag, lets use its frequency
    if pos in context_data[word]['POS']:
        freq=context_data[word]['POS'][pos]['freq']
    else:
        freq=0 
    
    #lets look if an inflection of the found word is MORE frequent
    if use_inflections:
        for inflection in context_data[word]['all_inflections']:
            if len(context_data[word]['all_inflections'])==1:
                continue
            base_pos=pos.split('_')[0] #VERBs can also be VERB_Past, VERB_PresT,...
            if pos in context_data[inflection]['POS']:
                infl_freq=context_data[inflection]['POS'][pos]['freq']
                if infl_freq>freq: #an inflection is (much) more common than the word
                    freq=infl_freq
                    word=inflection
            elif base_pos in context_data[inflection]['POS']:
                infl_freq=context_data[inflection]['POS'][base_pos]['freq']
                if infl_freq>freq: #an inflection is (much) more common than the word
                    freq=infl_freq
                    word=inflection
                    pos=base_pos
    return word,pos,freq

def get_context_util_nopos(word,pos,context_data,use_inflections=True):
    
    if word not in context_data:
        #current word has been seen 0 times in the whole corpus
        #and is not an inflection of any known word in the corpus
        return None,None,0
    freq=context_data[word]['freq']
    if use_inflections:
        #lets look if an inflection of the found word is MORE frequent
        for inflection in context_data[word]['all_inflections']:
            if len(context_data[word]['all_inflections'])==1:
                continue
            infl_freq=context_data[inflection]['freq']
            if infl_freq>freq: #an inflection is (much) more common than the word
                freq=infl_freq
                word=inflection

    pos=format_pos(word,pos)
    if pos not in context_data[word]['POS']:
        #find POS that is the most common
        most_common_pos=None
        most_common_pos_freq=0
        for pos in context_data[word]['POS']:
            pos_freq=context_data[word]['POS'][pos]['freq']
            if pos_freq>most_common_pos_freq:
                most_common_pos_freq=pos_freq
                most_common_pos=pos
        pos=most_common_pos
    return word,pos,freq

def get_context(context_data,sentence):
    
    pos_sentence=nltk.pos_tag(sentence.split(' '))
    min_freq=np.inf #words with high freq are considered as known
    min_freq_context=''
    min_freq_word=''
    min_freq_pos='UNK'
    #lets find the least frequent word in the sentence
    for word,pos in pos_sentence:
        word=word.lower()
        if len(word)<3:
            #one or two letter word are almost never informative
            continue
        word,pos,freq=get_context_util(word,pos,context_data)

        if freq==0:
            continue #no context exists for this one
         #along the sentence, lets keep only the least frequent word
        if freq>0 and freq<min_freq:
            min_freq=freq
            min_freq_word=word
            min_freq_pos=pos

    if min_freq_word!='':
        min_freq_context=context_data[min_freq_word]['POS'][min_freq_pos]['context']
        assert min_freq_word in min_freq_context.lower(),(min_freq_word,min_freq_context)
        #get a sentence centered on the target word
        min_freq_context=format_context(min_freq_context,min_freq_word,max_sentence_len=25)
        min_freq_context='( '+min_freq_context+' )'
    return min_freq_context,min_freq_word,min_freq

def pretty_print(success,all_pairs,verbose=False):
    cout=[]
    all_successes=0
    all_pairs_value=0
    for bin in success:
        if all_pairs[bin]>0:
            res=round(success[bin]/all_pairs[bin],2)
            std=round(np.sqrt(res*(1-res)/all_pairs[bin]),3)
            all_successes+=success[bin]
            all_pairs_value+=all_pairs[bin]
            if verbose:
                cout.append(' '.join([str(bin),str(res),str(std),str(all_pairs[bin])]))
            else:
                cout.append(str(res))
    print('\n'.join(cout))
    res=all_successes/(0.00001+all_pairs_value)
    std=round(np.sqrt(res*(1-res)/(0.00001+all_pairs_value)),3)
    print('overall mean:',round(res,3),'+/-',str(std),all_pairs_value)
    cout.append(str(round(res,3)))
    return cout

def pretty_print_avg(success,all_pairs,verbose):
   
    verbose_cout,matrix,variance={},{},{}
    for bin in success:
        for pos in success[bin]:
            if all_pairs[bin][pos]>0:
                if bin not in matrix:
                    matrix[bin]={}
                    variance[bin]={}
                    verbose_cout[bin]=[] 
                if success[bin][pos]>0:
                    tmp_res=float(success[bin][pos])/float(all_pairs[bin][pos])
                    tmp_std=np.sqrt(tmp_res*(1-tmp_res)/all_pairs[bin][pos])
                    matrix[bin][pos]=np.around(tmp_res,4)
                    variance[bin][pos]=np.around(tmp_std*tmp_std,4)
                if verbose:
                    verbose_cout[bin].append(' '.join((pos,str(round(tmp_res,2)),str(round(tmp_std,2)),str(all_pairs[bin][pos]))))
    bins=[k for k in success.keys()]
    bins.sort() #need to add keys in order
    pos_tags=[k for k in matrix[list(matrix.keys())[0]].keys()]
    pos_tags.sort()
    
    sorted_matrix=[]
    for bin in bins:
        sorted_matrix.append([])
        for pos in pos_tags:
            if bin in matrix and pos in matrix[bin]:
                sorted_matrix[bin].append(matrix[bin][pos])
            else:
                sorted_matrix[bin].append(0)
    sorted_matrix=np.array(sorted_matrix)
    sorted_matrix[sorted_matrix == 0] = np.nan
    avg_per_bin=np.around(np.nanmean(sorted_matrix,axis=1),3)
    avg_per_pos=np.around(np.nanmean(sorted_matrix,axis=0),3)
    avg_per_bin_lt=np.around(np.nanmean(sorted_matrix[:4,:],axis=1),3)
    avg_per_pos_lt=np.around(np.nanmean(sorted_matrix[:4,:],axis=0),3)
   
   
    out={'AVG_BIN':list(avg_per_bin)}
    out['AVG_POS']=list(avg_per_pos)
    out['AVG']=np.around(np.nanmean(avg_per_bin),3)
    out['AVG_BIN_LT']=list(avg_per_bin_lt)
    out['AVG_POS_LT']=list(avg_per_pos_lt)
    out['AVG_LT']=np.around(np.nanmean(avg_per_pos_lt),3)
    out['BINS']=list(bins)
    out['POS']=list(pos_tags)
    out['MATRIX']=matrix
    if verbose:
        cout=[]
        cout.append(str(out))
        for bin in verbose_cout:
            cout.append(' '.join(verbose_cout[bin])) 
        cout.append(' '.join([str(b) for b in bins])) 
        cout.append(' '.join(pos_tags))       
        cout.append(' '.join(('AVG_BIN:',str(avg_per_bin),'AVG:',str(np.around(np.nanmean(avg_per_bin),3)))))
        cout.append(' '.join(('AVG_POS:',str(avg_per_pos),'AVG:',str(np.around(np.nanmean(avg_per_pos),3)))))
        cout.append(' '.join(('AVG_BIN_LT:',str(avg_per_bin_lt),'AVG_LT:',str(np.around(np.nanmean(avg_per_bin_lt),3)))))
        cout.append(' '.join(('AVG_POS_LT:',str(avg_per_pos_lt),'AVG_LT:',str(np.around(np.nanmean(avg_per_pos_lt),3)))))
        print('\n'.join(cout))
    else:
        cout=out 
        print(cout)
    return cout

def check_capital_and_punc(w1,w2,g1,g2,ig1,ig2):
    #word are lower case and sentences start with upper case
    w1=w1.lower()
    w2=w2.lower()
    g1=g1[0].upper()+g1[1:]
    g2=g2[0].upper()+g2[1:]
    #adding period at the end
    if g1[-1] not in ['.','!','?']:
        g1=g1+' .'
    if g2[-1] not in ['.','!','?']:
        g2=g2+' .'    
    if g1[0] in [',',' '] or g2[0] in [',',' ']:
        return None,None,None,None

    assert g1[-1] in ['.','!','?'],g1
    assert g2[-1] in ['.','!','?'],g2
    
    #there is no space in the sentences
    split_g1=g1.split(' ')
    split_g2=g2.split(' ')
    split_g1 = list(filter(None, split_g1))
    split_g2 = list(filter(None, split_g2))

    #the word are still in place
    assert ig1<len(split_g1),(split_g1,w1,ig1,split_g2,w2,ig2)
    assert ig2<len(split_g2),(split_g1,w1,ig1,split_g2,w2,ig2)
    assert split_g1[ig1].lower()==w1,(split_g1,w1,ig1,split_g2,w2,ig2)
    assert split_g2[ig2].lower()==w2,(split_g1,w1,ig1,split_g2,w2,ig2)
    g1=' '.join(split_g1)
    g2=' '.join(split_g2)
    return w1,w2,g1,g2