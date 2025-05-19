import numpy as np
import random, json
import tqdm

def make_prompt_minimal(word,inflection,pos,rule):
    pos=pos.lower()+'s'
    adding=''
    if rule=='VERB':
        adding='that each uses one of these two verbs'
    elif rule=='ADJ':
        adding='that each uses one of these two adjectives'
    elif rule=='NOUN':
        adding='that each uses one of these two nouns'
    elif rule=='DET':
        adding=' '.join(('that shows a determiner-noun agreement, using either that/these/this/those. For instance, using the nouns \'misconduct\' and \'misconducts\', you can write something like: \'This misconduct is a serious offense. These misconducts are serious offenses.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
    elif rule=='SVSHORT':
        adding='that show a short distance subject-verb agreement at the present tense. The subject and the verb must be placed close to each other'
    elif rule=='SVLONG':
        if pos=='VERB':
            adding=' '.join(('that shows a long distance subject-verb agreement through a relative clause starting by \'that can be\'. For instance, using the verbs \'let\' and \'lets\', you can write something like: \'The person that can be trusted lets the dog out. The people that can be trusted let the dog out.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
        else:
            adding=' '.join(('that shows a long distance subject-verb agreement through a relative clause starting by \'that can be\'. For instance, using the nouns \'neighbor\' and \'neighbors\', you can write something like: \'The neighbor that can be trusted lets his dog out. The neighbors that can be trusted let their dog out.\'. Now please do the same with \'',word,'\' and,\'',inflection,'\''))
    elif rule=='ANAPHORASHORT':
        adding=' '.join(('that shows a short distance usage of reflexive pronouns. The pronouns must be placed close to the subject \'',word,'\' and,\'',inflection,'\'. Please use the past tense'))
    elif rule=='ANAPHORALONG':
        adding=' '.join(('that shows a long distance usage of reflexive pronouns through a relative clause starting by \'that can be\'. For instance, using the verbs \'medecine\' and \'medecines\', you can write something like: \'The medecine that can be bought anywhere, proved itself to be very effective. The medecines that can be bought anywhere, proved themselves to be very effective.\'.  Now please do the same with \'',word,'\' and,\'',inflection,'\'')) 
    else:
        assert False,rule

    prompt=' '.join(("Using the",pos,"\'",word,"\' and,\'",inflection,"\', please write a minimal pair of sentences",adding,". You must encapsulate the two sentences together in between brackets."))
    return prompt

if __name__ == '__main__':
    #find all inflections that never appear in the training set.
    cuda=True
    corpus='100M'
    pair_file='BabyLM_2024_formatted/compositionality_task_'+corpus+'_filtered.txt'
    output_file='BabyLM_2024_formatted/minimal_ltblimp_'+corpus+'_prompts'
    output_file2='BabyLM_2024_formatted/inflectionswap_'+corpus+'_prompts'
    minimal_pairs=True
    with open(pair_file) as buf:
        lines=buf.readlines()
    out,out2=[],[]
    pairs={}
    freq_bins=np.array([0,1,2,4,8,16,32,64,128,256,np.inf])
    success={}
    for i in range(len(freq_bins)):   
        success[i]={}
        for rule in ['VERB','SVSHORT','SVLONG','ANAPHORASHORT','ANAPHORALONG','NOUN','ADJ','DET']:
            if rule not in success[i]:
                success[i][rule]=0
    c=0
    for line in tqdm.tqdm(lines):
        if len(line.rstrip().split('|'))!=7:
            continue
        bin,word,inflection,pos,sentence,gen1,gen2=line.rstrip().split('|')
        if word==inflection:
            continue
        if len(gen1)>1 and gen1[1:-1].lower()=='yes':
            #word and inflection are indeed two inflections of the same word
            prompt=make_prompt_minimal(word,inflection,pos,pos)
            success[int(bin)][pos]+=1   
            out2.append('|'.join((bin,word,inflection,pos,sentence,prompt)))
            out2.append('|'.join((bin,word,inflection,pos,sentence,prompt)))
            out2.append('|'.join((bin,word,inflection,pos,sentence,prompt)))
            out2.append('|'.join((bin,word,inflection,pos,sentence,prompt)))
            if (word[-1]!='s' and inflection[-1]!='s'):
                #if the pair is singular/plural nouns or first/third-person verbs
                #then it qualifies for SV agreements and ANAPHORE agreements (only for nouns)
                continue
            if len(gen2)>1 and gen2[1:-1].lower()=='yes':
                #if word can also be the subject of a sentence
                assert pos=='NOUN'
                rule='SVSHORT'
                prompt=make_prompt_minimal(word,inflection,pos,rule)
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                success[int(bin)][rule]+=1   
                rule='SVLONG'
                prompt=make_prompt_minimal(word,inflection,pos,rule)
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                success[int(bin)][rule]+=1   
                rule='ANAPHORALONG'
                prompt=make_prompt_minimal(word,inflection,pos,rule)
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                success[int(bin)][rule]+=1  
                rule='ANAPHORASHORT'
                prompt=make_prompt_minimal(word,inflection,pos,rule)
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                success[int(bin)][rule]+=1 
                rule='DET'
                prompt=make_prompt_minimal(word,inflection,pos,rule)
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                out.append('|'.join((bin,word,inflection,rule+'-'+pos,sentence,prompt)))
                success[int(bin)][rule]+=1  

    for bin in success:
        print(bin,success[bin])
    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')
    with open(output_file2,'w') as buf:
        buf.write('\n'.join(out2)+'\n')
    print(output_file,len(out))