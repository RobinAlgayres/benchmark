import os,sys

#llm prompt to create alternative sentences.
cluster_file=sys.argv[1]#'BabyLM_2024_formatted/freq_bins_100M'
prompt_file=sys.argv[2]#'BabyLM_2024_formatted/freq_bins_100M_prompts'


with open(cluster_file) as buf:
    lines=buf.readlines()

prompts=[]
for line in lines:
    bin,word,original_pos,_,_,index,sentence=line.rstrip().split('/')
    if sentence[-1]!='.':
        sentence=sentence+' .'
    if sentence[0].islower():
        sentence=sentence[0].upper()+sentence[1:]
    pos=original_pos.split('_')[0].lower()
    if pos=='adj':
        pos='adjective'
    elif pos=='ne':
        pos='proper noun'
    #tmp=("Given the",pos,"\'",word,"\' and given the sentence (start of sentence) ",sentence,"(end of sentence) Can you write a new simple sentence that contains the",pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets.")
    tmp=("Given the",pos,"\'",word,"\'.Can you write a simple sentence that contains the",pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets.")
    tmp=' '.join(tmp)
    prompts.append('|'.join((bin,word,original_pos,index,sentence,tmp)))

print(prompt_file)
with open(os.path.join(prompt_file),'w') as buf:
    buf.write('\n'.join(prompts)+'\n')




    

    
