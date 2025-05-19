#!/bin/bash

numberwords='10M'

python format_text.py #will produce a train_100M and train_10M filtered datasets



#compute the frequency and POS of all words and retrieve the longest sentence that contain that word
#POS are simplified to fit a small number of categories
dataset='BabyLM_2024_formatted/all_'$numberwords
dataset_raw='BabyLM_2024_formatted/all_'$numberwords'_raw'
longtail='BabyLM_2024_formatted/longtail_'$numberwords
dictionnary='BabyLM_2024_formatted/dictionnary_'$numberwords
python get_hist.py $dataset $longtail
python make_dictionnary $dataset_raw $dictionnary

# This script outputs the list of words+POS separated in bins. Each word+POS verifies that
# The sum of all other inflections and POS (i.e the cluster frequency) also belongto the same bin
# This mean these words qre particularly stable: they are mostly seen in that form and that POS.
# Their other forms and other POS are much less frequent
frequency_file='BabyLM_2024_formatted/freq_bins_'$numberwords
python cluster.py $longtail $frequency_file


#write prompts for each word, asking an llm to generate a sentence that contains that word
llm_prompts='BabyLM_2024_formatted/prompts_'$numberwords
python make_llm_prompts.py

#run new_sentences in Bento Next to generate the sentences using 
#this script returns: freq_bins_10M_generations

#make dictionnary by listing all word in dataset
generations='BabyLM_2024_formatted/freq_bins_'$numberwords'_generations'
pair_file='BabyLM_2024_formatted/pairs_'$numberwords


#create the pairs by checking that generated sentences only contains word present in the original dataset
python filter_llm_first_pass.py

#create the prompts to check if words have enough context in the dataset to be understood
pair_prompt='BabyLM_2024_formatted/pairs_10M_prompts'
python pair_llm_prompts.py $pair_file $pair_prompts

#run prompt_rejects.py in Bento Next

