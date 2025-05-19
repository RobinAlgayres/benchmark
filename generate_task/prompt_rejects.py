import asyncio
import os

import metagen.bento
import tqdm
from metagen import (
    ChatRole,
    ClassifierResponse,
    CompletionResponse,
    Message,
    MetaGenKey,
    MetaGenPlatform,
)
'''
"mg-api-e758555775e0",
                    "mg-api-9e66a7b52a0e",
                    "mg-api-8bea88a77d16",
                    "mg-api-3594d63e7150",
                    "mg-api-4df0a1308e04",
'''

#metagen_key = "mg-api-91265418ad95" #my metagen key
metagen_key = "mg-api-e758555775e0" #for llama405b-fp8

metagen_platform: MetaGenPlatform = metagen.bento.create_metagen_platform(
    MetaGenKey(key=metagen_key), auto_rate_limit=True
)

output_dir = "/home/robinalgayres/prompt_rejects"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# pyre-ignore
def filter_gen(response):
    gen = response.choices[0].text
    if "[" not in gen or "]" not in gen:
        return "[]"
    begin = gen.find("[")
    end = gen.find("]")
    gen = gen[begin + 1 : end]
    return gen


async def completion_job_with_print_statement_async(lines, i) -> None:
    # model = "llama3.3-70b-instruct"
    model = "llama3-405b-choo_choo_fp8_annotations"
    line = lines[i].rstrip().split("|")
    (
        prompt1,
        prompt11,
        prompt2,
        prompt22,
    ) = line[-8:-4]
    answers = []
    # asking four times the llm for each prompt to avoid errors
    for prompt in [prompt1, prompt11, prompt2, prompt22]:
        formatted_prompt = "\n".join(prompt.split("_"))
        response: CompletionResponse = await metagen_platform.chat_completion_async(
            model=model,
            messages=Message.message_list().add_user_message(formatted_prompt).build(),
            temperature=0,
            # top_p=0.9,
        )
        gen = filter_gen(response)
        line.append(gen)
        answers.append(gen)
    line = "|".join(line)
    return line


with open("/home/robinalgayres/pairs_10M_prompts") as buf:
    lines = buf.readlines()
# Create jobs and run them in parallel with asyncion.gather. What you will see is
# that the print statements will start printing out numbers as completions finalize, at a
# rate that matches this MetaGen key's rate limit configuration
# pyre-fixme[76]: `await` may only be used inside an async definition.
buffer = 100
nb_lines = len(lines)
nb_batch = int(nb_lines / buffer) + 1
for b in range(nb_batch):
    output_file = os.path.join(output_dir, str(b))
    if os.path.isfile(output_file):
        print(output_file, "already exists")
        continue
    start = b * buffer
    end = (b + 1) * buffer
    try:
        output = await asyncio.gather(
            *[
                completion_job_with_print_statement_async(lines, i)
                for i in range(start, end)
            ]
        )
    except Exception as e:
        print(e, b)
        continue
    # print(output)
    print(b, start, end, len(output), os.path.join(output_dir, str(b)))
    with open(output_file, "w") as buf:
        buf.write("\n".join(output) + "\n")

import glob
import shutil
nb_l=0
with open("/home/robinalgayres/pairs_100M_prompts_new_file", "wb") as outfile:
    for filename in glob.glob(output_dir + "/*"):
        with open(filename, "rb") as readfile:
            shutil.copyfileobj(readfile, outfile)
        with open(filename, "rb") as readfile:
            nb_l+=len(readfile.readlines())

print('finish, number lines',nb_l)