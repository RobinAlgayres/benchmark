# LongTail-Swap: benchmarking language models’ abilities on rare words.

This repository contains:

1- Code to create the LT-Swap tasks on a given text dataset. 

2- LT-Swap10M and LT-Swap100M constructed on the 10M and 100M version of the BabyLM datasets

## Installation

Get the latest version of transformers, nltk and pytorch

## Creating tasks from a pretraining set

The whole process is identical for any English text dataset. As an example we set $PRETRAINING_DIR to the BabyLM 10M words text datasets (from https://babylm.github.io/). And we set $TASK_DIR to the task directory to store the future task. 

First step is to create a list of word candidates and word inflections for each file in the BabyLM datasets. This is done in parallel with 5 cpus.
```
python generate_task/get_word_lists.py  --data=$PRETRAINING_DIR/ --output_wordlists_dir=$TASK_DIR/wordlists/ --ncpus=5
```
We merge those words lists and create two files: one list of words for WordSwap and one list of inflected pairs for InflectionSwap and AgreementSwap
```
python generate_task/build_longtail.py --wordlists_dir=$TASK_DIR/wordlists/ --output_wordlist=$TASK_DIR/longtail_wordlist --output_inflpairs=$TASK_DIR/longtail_inflpairs --output_voc=$TASK_DIR/vocabulary
```

### Creating WordSwap

This prompts will directly ask an LLM to generate sentences for WordSwap.
```
python generate_task/wordswap_sentence_prompts.py --wordlist=$TASK_DIR/longtail_wordlist --output_file=$TASK_DIR/wordswap_sentence_prompts
```
The prompts are sent to the LLM for generation
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py --input_file=$TASK_DIR/wordswap_sentence_prompts --output_dir=$TASK_DIR/tmp_dir --output_file=$TASK_DIR/wordswap_sentence_generations
rm -r $TASK_DIR/tmp_dir
```
Create the prompts for the LLM filtering step
```
python generate_task/wordswap_pairs_and_filtering_prompts.py --input_file=$TASK_DIR/wordswap_sentence_generations --output_file=$TASK_DIR/wordswap_sentence_pairs_filtering_prompts --voc_file=$TASK_DIR/vocabulary
```
The prompts are sent to the LLM for filtering. The output is the final WordSwap task: a text file with the sentence pairs that passed the filter.
```
mkdir $TASK_DIR/tmp_dir
python generate_task/hf_generate.py --input_file=$TASK_DIR/wordswap_sentence_pairs_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/wordswap_sentence_pairs
rm -r $TASK_DIR/tmp_dir
```
### Creating InflectionSwap and AgreementSwap

For InflectionSwap and AgreementSwap we first ask an LLM if the automatically computed inflected pairs are indeed inflections of the same word. In addition for AgreementSwap we ask the LLM if the inflected pairs are words that could take a reflexive pronoun.
```
python generate_task/inflpairs_filtering_prompts.py --inflpairs=$TASK_DIR/longtail_inflpairs --output_file=$TASK_DIR/inflpairs_filtering_prompts
```
The prompts are sent to the LLM for filtering
```
mkdir $TASK_DIR/tmp_dir
python generate_task/hf_generate.py $TASK_DIR/syntax_words_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_words_to_be_filtered
rm -r $TASK_DIR/tmp_dir
```
Filter words and create the prompts for AgreementSwap and InflectionSwap sentence generations
```
python generate_taskssyntax_sentence_prompts.py $TASK_DIR/syntax_words_to_be_filtered $TASK_DIR/syntax_sentence_pairs_prompts
```
The prompts are sent to the LLM for generations
```
mkdir $TASK_DIR/tmp_dir
python generate_task/hf_generate.py $TASK_DIR/syntax_sentence_pairs_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_sentence_pairs_generations
rm -r $TASK_DIR/tmp_dir
```

Create the prompts for the LLM filtering step
```
python generate_task/syntax_sentence_pairs_filtering_prompts.py $TASK_DIR/syntax_sentence_generations $TASK_DIR/syntax_sentence_pairs_filtering_prompts
```
The prompts are sent to the LLM for filtering. The output is the final Agreement and InflectionSwap tasks: a text file with the sentence pairs that passed the filter. 
```
mkdir $TASK_DIR/tmp_dir
python generate_task/hf_generate.py $TASK_DIR/syntax_sentence_pairs_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_sentence_pairs
rm -r $TASK_DIR/tmp_dir
```

## Evaluating LM on LT-Swap10M and LT-Swap100M

LT-Swap10M and LT-Swap100M are created based on the BabyLM 10M and 100M words text datasets. Each task is composed of three subtask files for WordSwap, InflectionSwap and AgreementSwap. Each line is formatted as follows:
```
<frequency bin index> <POS tag> <target word 1> <generated sentence 1> <index of word 1 in sentence 1> <target word 2> <generated sentence 2> <index of word 2 in sentence 2>
```
In order to evaluate a model on any subtasks do the following
```
python eval_longtail.py <path to subtasks file> <huggingface model name>
```
