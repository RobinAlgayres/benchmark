# LongTail-Swap: benchmarking language models’ abilities on rare words.

This repository contains:

1- Code to create the LT-Swap tasks on a given text dataset. 

2- LT-Swap10M and LT-Swap100M constructed on the 10M and 100M version of the BabyLM datasets

### Installation

Get the latest version of transformers, nltk and pytorch

### Creating tasks from a pretraining set

The whole process is identical for any English text dataset. As an example we set $PRETRAINING_DIR to the BabyLM 10M words text datasets (from https://babylm.github.io/). And we set $TASK_DIR to the task directory to store the future task. 

First step is to create a list of word candidates and word inflections for each file in the BabyLM datasets. This is done in parallel with 5 cpus.
```
python generate_tasks/get_words_lists.py  $PRETRAINING_DIR/ $TASK_DIR/wordlist_per_file/ 5
```
We merge those words lists and create two files: one list of words for WordSwap and one list of inflected pairs for InflectionSwap and AgreementSwap
```
python generate_tasks/build_longtail.py $TASK_DIR/wordlist_per_file/ $TASK_DIR/longtail_semantic $TASK_DIR/longtail_syntax
```
We create the prompts for the first LLM phase. These prompts will directly generate sentences for WordSwap. For InflectionSwap and AgreementSwap we first ask an LLM if the automatically computed inflected pairs are indeed inflections of the same word. In addition for AgreementSwap we ask the LLM if the inflected pairs are words that could take a reflexive pronoun.
```
python generate_tasks/phase_one_prompts.py $TASK_DIR/longtail_semantic $TASK_DIR/longtail_syntax $TASK_DIR/phase_one_prompts
```
The prompts are sent to the LLM of your choice (we advise here to use one as big as possible)
```
mkdir $TASK_DIR/tmp_dir
python generate_tasks/hf_generate.py $TASK_DIR/phase_one_prompts $TASK_DIR/tmp_dir $TASK_DIR/phase_one_generations
rm -r $TASK_DIR/tmp_dir
```

### LT-Swap10M and LT-Swap100M

TODO
