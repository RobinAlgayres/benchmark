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
python generate_tasks/wordswap_sentence_pairs_and_filtering_prompts.py $TASK_DIR/wordswap_sentence_generations $TASK_DIR/wordswap_sentence_pairs_filtering_prompts
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
The prompts are sent to the LLM of your choice (we advise here to use one as big as possible)
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntax_words_filtering_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_words_filtered
rm -r $TASK_DIR/tmp_dir
```
Create the prompts for AgreementSwap and InflectionSwap sentence generations
```
python generate_tasks/syntax_sentence_prompts.py $TASK_DIR/syntax_words_filtered $TASK_DIR/syntax_sentence_prompts
```
The prompts are sent to the LLM of your choice
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntax_sentence_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntax_sentence_generations
rm -r $TASK_DIR/tmp_dir
```

Retrieve the feasable pairs of WordSwap sentences. The output file is the final WordSwap task, please refer to the evaluation section for usage of the task.
```
python retrieve_correct_pairs.py $TASK_DIR/wordswap_pairs_to_be_filtered $TASK_DIR/wordswap_pairs
```

```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/syntaxtasks_prompts $TASK_DIR/tmp_dir $TASK_DIR/syntaxtasks_generations
rm -r $TASK_DIR/tmp_dir
```
Filter LLM generations for the three tasks and create the last filtering prompts
```
python generate_tasks/phase_three_prompts.py $TASK_DIR/phase_one_generations $TASK_DIR/syntaxtasks_prompts
```

### LT-Swap10M and LT-Swap100M

TODO
