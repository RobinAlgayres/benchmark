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

#model_name = "huggingface/llama-7b"  # Replace with the model name you want to use
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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


def filter_gen(gen):
    if "[" not in gen or "]" not in gen:
        return "[]"
    begin = gen.rfind("[")
    end = gen.rfind("]")
    gen = gen[begin + 1 : end]
    return gen

# Async wrapper for running blocking code in a separate thread
async def generate_completion(line):
    line = line.rstrip().split("|")
    if len(line)==6:
        prompts = line[-1]
        temperature = 0.5
        use_filter = False
    elif len(line) == 20:
        prompts = line[-8:-4]
        temperature = 0
        use_filter = True
    else:
        print("wrong format", line)
        assert False, line
    print(prompts)
    for prompt in prompts:
        formatted_prompt = "\n".join(prompt.split("_"))
        inputs = tokenizer(formatted_prompt, return_tensors="pt")  # Tokenize the prompt
        outputs = await model.generate(inputs.input_ids, max_length=200, num_return_sequences=1,temperature=temperature)
        gen= tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(gen)
        gen = (
            gen.text.replace("\r\n", " ")
            .replace("\n", " ")
            .replace("\r", " ")
            )
        if use_filter:
            gen = filter_gen(gen)
        line.append(gen)
    line = "|".join(line)
    return line

# Async wrapper to run the blocking function in a thread
async def async_generate_completion(prompt):
    return await asyncio.to_thread(generate_completion, prompt)

# Run all prompts asynchronously
async def generate_completions(prompts):
    tasks = [async_generate_completion(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

async def generate(lines,start,end):
    output = await generate_completions(lines[start:end])
    #output= await asyncio.gather(*[async_generate_completion(lines[i]) for i in range(start, end)])
    return output

# Run the async code
if __name__ == "__main__":
    # Load the tokenizer and model from Hugging Face (this part is synchronous)
    input_file = sys.argv[1] #tasks_10M/wordswap_generation_prompts 
    output_dir = sys.argv[2] #tasks_10M/tmp_dir
    final_output_file = sys.argv[3] #task_10M/wordswap_generations
    # shutil.rmtree(output_dir)
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
            output = generate(lines,start,end)
            print(output)
        except Exception as e:
            #print(e, b)
            print(b, "failed")
            continue
        print(b, start, end, len(output), os.path.join(output_dir, str(b)))
        with open(output_file, "w") as buf:
            buf.write("\n".join(output) + "\n")
    #concat all generated files in one
    get_files(output_dir,final_output_file)