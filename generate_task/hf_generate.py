import asyncio
import os
import shutil
import string
import glob
import shutil,sys
import numpy
import tqdm
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse


model_name = "arnir0/Tiny-LLM" # Replace with the model name you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_files(output_dir,final_output_file):
    out = []
    for filename in os.listdir(output_dir):
        if filename[0] == ".":
            continue
        fpath = os.path.join(output_dir, filename)
        with open(fpath, "r") as readfile:
            lines = readfile.readlines()
            for line in lines:
                out.append(line.rstrip())

    print(len(out))
    print(final_output_file)
    with open(final_output_file, "w") as outfile:
        outfile.write("\n".join(out) + "\n")

def format_answer(g):
    start = g.rfind("[")
    end = g.rfind("]")
    if start == -1 or end == -1:
        if g in ["A", "B"]:
            return g
        else:
            return None
    g = g[start + 1 : end]
    g = g.replace(" ", "")
    return g.upper()


def correct_answers(llm_answers, answers):
    ag1, ag11, ag2, ag22 = llm_answers
    a1, a11, a2, a22 = answers
    ag1 = format_answer(ag1)
    ag11 = format_answer(ag11)
    ag2 = format_answer(ag2)
    ag22 = format_answer(ag22)
    if ag1 == a1 and ag11 == a11 and ag2 == a2 and ag22 == a22:
        return True
    else:
        return False
# Async wrapper for running blocking code in a separate thread
async def generate_completion(arg):
    line,temperature=arg
    line = line.rstrip().split("|")
    if "/" in line[-1]:
        # when several prompts and answers
        line, prompts_answers = line[:-1], line[-1]
        prompts_answers = prompts_answers.split("/")
        ind = int(len(prompts_answers) / 2)
        prompts, answers = prompts_answers[:ind], prompts_answers[-ind:]
    else:
        line, prompt = line[:-1], line[-1]
        prompts = [prompt]

    llm_answers = []
        
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")  # Tokenize the prompt
        outputs = await model.generate(inputs.input_ids, max_length=200, num_return_sequences=1,temperature=temperature)
        gen= tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen = (
            gen.text.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            )
        llm_answers.append(gen)

    if len(llm_answers) > 1:
        # using LLM for sentence filtering
        keep_line = correct_answers(llm_answers, answers)
        if not keep_line:
            return None
    else:
        # using LLM for sentence generation
        line.extend(llm_answers)
    line = "|".join(line)  # prompt removed and generations added to initial line
    return line

# Async wrapper to run the blocking function in a thread
async def async_generate_completion(arg):
    return await asyncio.to_thread(generate_completion, arg)

# Run all prompts asynchronously
async def generate_completions(args):
    tasks = [async_generate_completion(arg) for arg in args]
    return await asyncio.gather(*tasks)

async def generate(lines,start,end,temperature):
    batch=[]
    for line in lines[start:end]:
        batch.append((line,temperature))
    output = await generate_completions(batch)
    #output= await asyncio.gather(*[async_generate_completion(lines[i]) for i in range(start, end)])
    return output

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",type=str,help='path to file with prompts',required=True)
    parser.add_argument("--output_dir",type=str,help='path to tmp dir to store llm outputs',default='babylm-lt-swap/tmp_files_10M/')
    parser.add_argument("--output_file",type=str,help='path to output file with llm outputs',required=True)
    parser.add_argument("--temp",type=float,help='temperature, tune this parameter depending on your LLM typically low temp for filtering and high temp for generation',default=0.0)
    return parser.parse_args(argv)


# Run the async code
if __name__ == "__main__":
    args=parse_arguments(sys.argv[1:])
    input_file =args.input_file
    output_dir =args.output_dir
    final_output_file = args.output_file
    temperature=args.temp

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(input_file) as buf:
        lines = buf.readlines()


    buffer = 2
    nb_lines = len(lines)
    nb_batch = int(nb_lines / buffer) + 1
    for b in range(nb_batch):
        start = b * buffer
        end = (b + 1) * buffer
        output_file = os.path.join(output_dir, str(b))
        if os.path.isfile(output_file):
            print(output_file, "already exists")
            continue
        try:
            # Create jobs and run them in parallel with asyncion.gather.
            # pyre-fixme[76]: `await` may only be used inside an async definition.
            output = generate(lines,start,end,temperature)
            print(output)
        except Exception as e:
            #print(e, b)
            print(b, "failed")
            continue
        output = [o for o in output if o is not None]
        print(b, start, end, len(output), os.path.join(output_dir, str(b)))
        with open(output_file, "w") as buf:
            buf.write("\n".join(output) + "\n")
    #concat all generated files in one
    get_files(output_dir,final_output_file)