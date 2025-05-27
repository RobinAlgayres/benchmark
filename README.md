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
python generate_tasks/get_words_lists.py  $PRETRAINING_DIR/ $TASK_DIR/wordlist_per_file/ 5
```
We merge those words lists and create two files: one list of words for WordSwap and one list of inflected pairs for InflectionSwap and AgreementSwap
```
python generate_tasks/build_longtail.py $TASK_DIR/wordlist_per_file/ $TASK_DIR/longtail_semantic $TASK_DIR/longtail_syntax
```

### Creating WordSwap

This prompts will directly ask an LLM to generate sentences for WordSwap.
```
python generate_tasks/wordswap_sentence_prompts.py $TASK_DIR/longtail_semantic $TASK_DIR/wordswap_sentence_prompts
```
The prompts are sent to the LLM for generation
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/wordswap_sentence_prompts $TASK_DIR/tmp_dir $TASK_DIR/wordswap_sentence_generations
rm -r $TASK_DIR/tmp_dir
```
Create the prompts for the LLM filtering step
```
python generate_tasks/wordswap_sentence_pairing_and_filtering_prompts.py $TASK_DIR/wordswap_sentence_generations $TASK_DIR/wordswap_sentence_pairs_filtering_prompts
```
The prompts are sent to the LLM for filtering
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/wordswap_sentence_pairs_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/wordswap_sentence_pairs_to_be_filtered
rm -r $TASK_DIR/tmp_dir
```
Retrieve the feasable pairs of WordSwap sentences. The output file is the final WordSwap task, please refer to the evaluation section for usage of the task.
```
python retrieve_correct_pairs.py $TASK_DIR/wordswap_sentence_pairs_to_be_filtered $TASK_DIR/wordswap_sentence_pairs
```
### Creating InflectionSwap and AgreementSwap

For InflectionSwap and AgreementSwap we first ask an LLM if the automatically computed inflected pairs are indeed inflections of the same word. In addition for AgreementSwap we ask the LLM if the inflected pairs are words that could take a reflexive pronoun.
```
python generate_tasks/syntax_words_filtering_prompts.py $TASK_DIR/longtail_syntax $TASK_DIR/syntax_words_filtering_prompts
```
The prompts are sent to the LLM for filtering
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntax_words_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_words_to_be_filtered
rm -r $TASK_DIR/tmp_dir
```
Filter words and create the prompts for AgreementSwap and InflectionSwap sentence generations
```
python generate_tasks/syntax_sentence_prompts.py $TASK_DIR/syntax_words_to_be_filtered $TASK_DIR/syntax_sentence_pairs_prompts
```
The prompts are sent to the LLM for generations
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntax_sentence_pairs_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_sentence_pairs_generations
rm -r $TASK_DIR/tmp_dir
```

Create the prompts for the LLM filtering step
```
python generate_tasks/syntax_sentence_pairs_filtering_prompts.py $TASK_DIR/syntax_sentence_generations $TASK_DIR/syntax_sentence_pairs_filtering_prompts
```
The prompts are sent to the LLM for generations
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntax_sentence_pairs_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_sentence_pairs_to_be_filtered
rm -r $TASK_DIR/tmp_dir
```
Retrieve the feasable pairs of WordSwap sentences. The output file is the final WordSwap task, please refer to the evaluation section for usage of the task.
```
python retrieve_correct_pairs.py $TASK_DIR/syntax_sentence_pairs_to_be_filtered $TASK_DIR/syntax_sentence_pairs
```

## Evaluating LM on LT-Swap10M and LT-Swap100M

LT-Swap10M and LT-Swap100M are created based on the BabyLM 10M and 100M words text datasets. Each task is composed of three files for WordSwap, InflectionSwap and AgreementSwap. Each line is formatted as follows:
```
<frequency bin index> <POS tag> <target word 1> <generated sentence 1> <index of word 1 in sentence 1> <target word 2> <generated sentence 2> <index of word 2 in sentence 2>
```

