import os
import metagen.bento
import tqdm
from metagen import CompletionResponse, Message, MetaGenKey, MetaGenPlatform
import asyncio
import glob
import shutil

metagen_key = "mg-api-91265418ad95"
#model = "llama3.1-70b-instruct"

metagen_platform: MetaGenPlatform = metagen.bento.create_metagen_platform(
    MetaGenKey(key=metagen_key), auto_rate_limit=True
)

output_dir = "/home/robinalgayres/all_new_10M"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
# pyre-ignore
async def completion_job_with_print_statement_async(lines, i) -> None:
    bin, word, pos, index, sentence = lines[i].rstrip().split("|")
    response: CompletionResponse = await metagen_platform.chat_completion_async(
        messages=Message.message_list().add_user_message(sentence).build(),
        max_tokens=200,
    )

    # The model will generate JSON output in the format of the schema above
    gen = response.choices[0].text
    if "[" not in gen or "]" not in gen:
        gen = "[empty]"
    begin = gen.find("[")
    end = gen.find("]")
    if end - begin > 10:
        gen = gen[begin : end + 1]
    output_gen = "|".join((bin, word, pos, index, sentence, gen))
    return output_gen


with open("/home/robinalgayres/all_10M") as buf:
    lines = buf.readlines()
# Create jobs and run them in parallel with asyncion.gather. What you will see is
# that the print statements will start printing out numbers as completions finalize, at a
# rate that matches this MetaGen key's rate limit configuration
# pyre-fixme[76]: `await` may only be used inside an async definition.
buffer = 100
nb_lines = len(lines)
nb_batch = int(nb_lines / buffer) + 1
for b in range(nb_batch):
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
    print(b, start, end, len(output), os.path.join(output_dir, str(b)))
    with open(os.path.join(output_dir, str(b)), "w") as buf:
        buf.write("\n".join(output) + "\n")

nb_l=0
with open("/home/robinalgayres/all_new_10M_file", "wb") as outfile:
    for filename in glob.glob(output_dir + "/*"):
        with open(filename, "rb") as readfile:
            shutil.copyfileobj(readfile, outfile)
        with open(filename, "rb") as readfile:
            nb_l+=len(readfile.readlines())

print('finish, number lines',nb_l)