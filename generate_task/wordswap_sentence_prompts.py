import tqdm
import sys
import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordslist",type=str,help='path to words list file for WordSwap',default='babylm-lt-swap/tmp_files_10M/longtail_wordslist')
    parser.add_argument("--output_file",type=str,help='path to sentence generation prompts file',default='babylm-lt-swap/tmp_files_10M/wordswap_sentence_prompts')
    return parser.parse_args(argv)

if __name__=='__main__':
    args=parse_arguments(sys.argv[1:])
    wordslist_file=args.wordslist
    output_file=args.output_file
    out=[]
    with open(wordslist_file) as buf:
        lines=buf.readlines()
    for line in tqdm.tqdm(lines):
        bin,word,pos,index,sentence=line.rstrip().split('|')
        base_pos=pos.split('_')[0].lower() 
        assert base_pos in ['noun','verb'],base_pos
        prompt=' '.join(("Given the",base_pos,"\'",word,"\'. Can you write a simple sentence that contains the",base_pos,"\'",word,"\' using at least 20 words. Make it simple. Write only this sentence between brackets."))
        out.append('|'.join((bin,word,pos,index,sentence,prompt)))
    
    with open(output_file,'w') as buf:
        buf.write('\n'.join(out)+'\n')