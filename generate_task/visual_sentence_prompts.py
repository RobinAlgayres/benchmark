import numpy as np
import random, json, sys
import tqdm, argparse


def make_visual_prompt_minimal(w1,w2):
    prompt = (
    f"Using the two words '{w1}' and '{w2}', create a minimal pair of sentences. "
    f"Each sentence should use one of these words making a reference to how "
    f"different those words are in their typical size, weight, or shape in the real world. "
    f"Encapsulate both sentences together within brackets."
    )
    return ''.join(prompt)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded_word_list",type=str,help='path to inflected pairs filtering generation file',default='visual_task/ground_word_list')
    parser.add_argument("--output_file",type=str,help='path to minimal sentence pairs prompts',default='visual_task/visual_sentence_prompts')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    grounded_word_list=args.grounded_word_list
    output_file=args.output_file
    words={}
    freq_bins=np.array([128,256,512,1024,2048,4096,8192,16384,np.inf])
    with open(grounded_word_list) as buf:
        data=json.load(buf)
        for item in data:
            word=item['noun']
            if word not in words:
                words[word]=0
            words[word]+=1
    keys=list(words.keys())
    visual_pairs=[]
    seen_visual_pairs=set()
    max_pairs_per_bin=2000
    for i in range(len(keys)):
        w1=keys[i]
        f1=words[w1]
        for j in range(i+1,len(keys)):
            w2=keys[j]
            f2=words[w2]
            if w1==w2:
                continue
            #if new pair add it to the output
            key=[w1,w2]
            key.sort()
            key='-'.join((w1,w2))
            if key in seen_visual_pairs:
                continue
            seen_visual_pairs.add(key)
            prompt=make_visual_prompt_minimal(w1,w2)
            f=min(f1,f2)
            bin=np.where(f>=freq_bins)[0][-1]
            visual_pairs.append('|'.join((str(bin),w1,w2,'NOUN','VISUAL',prompt)))
            visual_pairs.append('|'.join((str(bin),w1,w2,'NOUN','VISUAL',prompt)))
            visual_pairs.append('|'.join((str(bin),w1,w2,'NOUN','VISUAL',prompt)))
            visual_pairs.append('|'.join((str(bin),w1,w2,'NOUN','VISUAL',prompt)))
    print(output_file,len(visual_pairs))
    with open(output_file,'w') as buf:
        buf.write('\n'.join(visual_pairs)+'\n')
     
    