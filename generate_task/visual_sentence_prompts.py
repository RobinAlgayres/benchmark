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
    freq_bins=np.array([1,2,4,8,16,32,64,128,256,512,np.inf])
    with open(grounded_word_list) as buf:
        for line in buf:
            word,freq=line.rstrip().split(',')
            try:
                bin=np.where(int(freq)>=freq_bins)[0][-1]
            except:
                continue  
            assert word not in words
            words[word]=bin
                
    keys=list(words.keys())
    keys.sort()
    visual_pairs=[]
    seen_visual_pairs=set()
    max_pairs_per_bin=2000
    max_diff_between_bins=0 #if 0, bins word pairs belong to the same frequency bin, increase
                            #this value for small dataset
    pairs_per_bin={}

    for i in range(len(keys)):
        w1=keys[i]
        b1=words[w1]
        for j in range(i+1,len(keys)):
            w2=keys[j]
            b2=words[w2]
            if abs(b2-b1)>max_diff_between_bins:
                continue
            bin=min(b2,b1)
            assert w1!=w2
            #if new pair add it to the output
            key=[w1,w2]
            key.sort()
            key='-'.join((w1,w2))
            if key in seen_visual_pairs:
                continue
            seen_visual_pairs.add(key)
            if bin not in pairs_per_bin:
                pairs_per_bin[bin]=[]
            pairs_per_bin[bin].append((w1,w2))

    bins=list(pairs_per_bin.keys())
    bins.sort()
    print('saving pairs per bin:')
    for bin in bins:
        print('bin:',bin,'nb of pairs:',len(pairs_per_bin[bin]))
        random.shuffle(pairs_per_bin[bin])
        pairs=pairs_per_bin[bin][:max_pairs_per_bin]
        for w1,w2 in pairs:
            prompt=make_visual_prompt_minimal(w1,w2)
            visual_pairs.append('|'.join((str(bin),w1,w2,'NOUN','VISUAL',prompt)))
    print(output_file,'kept word pairs:',len(visual_pairs))
    with open(output_file,'w') as buf:
        buf.write('\n'.join(visual_pairs)+'\n')
     
    