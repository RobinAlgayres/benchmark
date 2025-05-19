#!/bin/bash


model_names_100M=(
            /datasets/pretrained-llms/Llama-3.2-1B/
            'babylm/babyllama-100m-2024'
            'babylm/opt-125m-strict-2023'
            'babylm/roberta-base-strict-2023'
            'babylm/ltgbert-100m-2024'
            'ltg/gpt-bert-babylm-base'
            'bg51717/antlm-bert-ntp_mlm-100m'
            'SzegedAI/babylm24_LSM_strict'
            'SzegedAI/babylm24_MLSM_strict'
            )
model_names_10M=(
            #/datasets/pretrained-llms/Llama-3.2-1B/
            #'babylm/babyllama-10m-2024'
            #'babylm/opt-125m-strict-small-2023'
            #'babylm/roberta-base-strict-small-2023'
            #'babylm/ltgbert-10m-2024'
            #'ltg/gpt-bert-babylm-small'
            #'bg51717/antlm-bert-ntp_mlm-10m'
            #'SzegedAI/babylm24_LSM_strict-small'
            #'SzegedAI/babylm24_MLSM_strict-small'
            )

blimp_100M='blimp/formatted_blimp_100M'
blimp_10M='blimp/formatted_blimp_10M'
wordswap_100M='BabyLM_2024_formatted/wordswap_pairs_100M_filtered'
wordswap_10M='BabyLM_2024_formatted/wordswap_pairs_10M_filtered'
inflswap_100M='BabyLM_2024_formatted/inflswap_pairs_100M'
inflswap_10M='BabyLM_2024_formatted/inflswap_pairs_10M'
syntax_100M='BabyLM_2024_formatted/syntax_pairs_100M'
syntax_10M='BabyLM_2024_formatted/syntax_pairs_10M'
mkdir -p logs

for model_name in "${model_names_100M[@]}"; do
    
    #echo $model_name $wordswap_100M 
    #sbatch --output logs/$model_name'.wordout' --error logs/$model_name'.worderr' --export=ARG1=$wordswap_100M,ARG2=$model_name launcher.sh
    #echo $model_name $inflswap_100M 
    #sbatch --output logs/$model_name'.inflout' --error logs/$model_name'.inflerr' --export=ARG1=$inflswap_100M,ARG2=$model_name launcher.sh
    #echo $model_name $syntax_100M 
    #sbatch --output logs/$model_name'.synout' --error logs/$model_name'.synerr' --export=ARG1=$syntax_100M,ARG2=$model_name launcher.sh
    echo $model_name $blimp_100M 
    sbatch --output logs/$model_name'.blimpout' --error logs/$model_name'.blimperr' --export=ARG1=$blimp_100M,ARG2=$model_name launcher.sh
    
done


for model_name in "${model_names_10M[@]}"; do
    #echo $model_name $wordswap_10M 
    #sbatch --output logs/$model_name'.wordout' --error logs/$model_name'.worderr' --export=ARG1=$wordswap_10M,ARG2=$model_name launcher.sh
    #echo $model_name $inflswap_10M 
    #sbatch --output logs/$model_name'.inflout' --error logs/$model_name'.inflerr' --export=ARG1=$inflswap_10M,ARG2=$model_name launcher.sh
    #echo $model_name $syntax_10M 
    #sbatch --output logs/$model_name'.synout' --error logs/$model_name'.synerr' --export=ARG1=$syntax_10M,ARG2=$model_name launcher.sh
    #echo $model_name $blimp_10M 
    #sbatch --output logs/$model_name'.blimpout' --error logs/$model_name'.blimperr' --export=ARG1=$blimp_10M,ARG2=$model_name launcher.sh
    
done
