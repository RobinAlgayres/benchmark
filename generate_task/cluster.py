import os,sys
import numpy as np
import ast
import tqdm 

# create clusters and find words for which cluster frequency 
# and word frequency fall in the same frequency bin.

#removing all words that are not noun verb or adj


def max_pos_tags(context_per_pos,word):
    max_freq=0
    max_freq_context=None 
    max_freq_pos=None
    max_sentence_len=100
    max_freq_index=None
    for pos in context_per_pos:
        
        split_sentence=context_per_pos[pos]['context'].split(' ')
        try:
            index=split_sentence.index(word)
        except:
            continue
        if context_per_pos[pos]['freq']>max_freq:
            max_freq=context_per_pos[pos]['freq']
            #getting a maximum sentence length
            if len(split_sentence)>max_sentence_len:
                start_ind=max(0,index-int(max_sentence_len/2))
                end_ind=min(len(split_sentence),index+int(max_sentence_len/2))
                missing_words=max_sentence_len-(end_ind-start_ind)
                if missing_words>0:
                    if start_ind==0:
                        end_ind+=missing_words
                    else:
                        start_ind-=missing_words
                split_sentence=split_sentence[start_ind:end_ind]
                assert word in split_sentence
                index=split_sentence.index(word)
                assert len(split_sentence)>(max_sentence_len/2)
                assert len(split_sentence)<=max_sentence_len+1
            max_freq_index=index
            max_freq_context=' '.join(split_sentence)
            max_freq_pos=pos
            assert split_sentence[index]==word
    return max_freq_context, max_freq, max_freq_pos,max_freq_index


if __name__=='__main__':
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,512,np.inf])
    freq_bin_words=np.array([[],[],[],[],[],[],[]])
    frequency_file=sys.argv[1] #'BabyLM_2024_formatted/longtail_100M'
    freqs={}
    postags={}
    clusters={}

    output_freq_bins_file=sys.argv[2] #'BabyLM_2024_formatted/freq_bins_100M'
    output_freq_bins=[]
    output_clusters=[]
    min_word_length=3
    with open(frequency_file) as buf:
        lines=buf.readlines()
    legal_characters='abcdefghijklmnopqrstuvwxyz'

    for line in tqdm.tqdm(lines):
        if '/' in line or '|' in line:
            continue
        data=ast.literal_eval(line)
        
        has_illegal_char=False
        for char in data['word'].lower():
            if char not in legal_characters:
                has_illegal_char=True
        if has_illegal_char:
            continue
        
        assert data['word'] not in freqs
        freqs[data['word']]=int(data['freq']) #this is total frequency across POS tags
        postags[data['word']]=data['POS']

        #removing word and inflections 1 or 2 letter long,
        #they will artificially bring the frequency of some word up
        if len(data['word'])<min_word_length:
            continue
        inflections=[infl for infl in list(data['all_inflections']) if len(infl)>=min_word_length]
        if len(inflections)==0:
            continue
        
        #sorting to ensure clusters do not repeat themselves
        inflections.sort()
        hash_value='-'.join(inflections)
        #mapping key to inflections
        if hash_value not in clusters:
            clusters[hash_value]=set()
            for inflection in inflections:
                clusters[hash_value].add(inflection)

    tmp={}
    for hash_value in clusters:
        cluster_line=[]
        #cluster_line=[' '.join(hash_value.split('-'))]
        total=0
        max_freq=0
        most_common_inflection=None
        #for each inflection in the cluster, getting the frequency, pos and sentence of most 
        #common pos tag for that inflection
        for inflection in clusters[hash_value]:
            freq=freqs.get(inflection,0)
            #cluster_line.append(str(freq))
            cluster_line.append(inflection+' '+str(freq))
            total+=freq #total freq among POS and inflections
            if freq>max_freq:
                sentence,most_common_freq,most_common_pos,most_common_index=max_pos_tags(postags[inflection],inflection)
                if sentence is None:
                    continue #only 'UNK' POS tag is available for that inflection
                max_freq=freq
                most_common_inflection=inflection
        if sentence is None: #only 'UNK' POS tag is available for that cluster
            continue   
        if len(most_common_inflection)>30 or len(sentence)==0:
            continue #word too long or not context
        total>0,hash_value
        cluster_line.append('|'+str(total))
        cluster_line=' '.join(cluster_line)

        cluster_freq_bin=np.where(total>=freq_bins)[0][-1]
        inflection_freq_bin=np.where(most_common_freq>=freq_bins)[0][-1]
        assert inflection_freq_bin>0
        # Checking that among all POS tags and all inflections for that word, the frequency of that word 
        # with that POS tag is roughly the same as the total frequency of its cluster. 
        # This word is particularly frequent in that inflection and POS tag.
        if cluster_freq_bin==inflection_freq_bin:
            if most_common_pos=='UNK':
                continue
            #no more than 500 element per bin and POS tag
            key=str(cluster_freq_bin)+'_'+most_common_pos.split('_')[0]
            if key not in tmp:
                tmp[key]=0
            #if tmp[key]>=200:
            #    continue
            tmp[key]+=1
            output_freq_bins.append('|'.join((str(cluster_freq_bin),most_common_inflection,most_common_pos,cluster_line,str(most_common_index),sentence)))
            
    keys=list(tmp.keys())
    keys.sort()
    for key in keys:
        print(key,tmp[key])  

    print(len(output_freq_bins))
    with open(output_freq_bins_file,'w') as buf:
        buf.write('\n'.join(output_freq_bins)+'\n')



