import os,sys
import tqdm
import numpy as np
pair_file=sys.argv[1] #'BabyLM_2024_formatted/pairs_10M_prompts_new_file.txt'
final_pair_file=sys.argv[2] #'BabyLM_2024_formatted/feasable_pairs_10M'

with open(pair_file) as buf:
    lines=buf.readlines()
correct_pairs=[]
pairs={}
accepted,accepted_pos,all_pairs,all_pairs_pos={},{},{},{}
h=[]
for i in tqdm.tqdm(range(len(lines))):
    #print(lines[i].rstrip().split('|'))
    try:
        bin,original_pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2,p1,p11,p2,p22,a1,a11,a2,a22,ag1,ag11,ag2,ag22=lines[i].rstrip().split('|')
    except:
        print(len(lines[i].split('|')))
        continue
    if w1==w2:
        continue
    h.append(int(bin))
    pos=original_pos.split('_')[0]
    if bin not in accepted:
        accepted[bin]=0
        all_pairs[bin]=0
        accepted_pos[bin]={}
        all_pairs_pos[bin]={}
    if pos not in accepted_pos[bin]:
        accepted_pos[bin][pos]=0
        all_pairs_pos[bin][pos]=0
    ag1,ag11,ag2,ag22=ag1.upper(),ag11.upper(),ag2.upper(),ag22.upper()
    #continue
    if a1==ag1 and a11==ag11:
        if a2==ag2 and a22==ag22:
            key=[w1,w2]
            key.sort()
            key='-'.join(key)
            if key in pairs:
                #print(i,w1,w2) #pair of word present twice
                pass
            else:
                pairs[key]=(g1,g2)
            correct_pairs.append('|'.join((bin,original_pos,w1,s1,i1,g1,ig1,w2,s2,i2,g2,ig2)))
            accepted[bin]+=1
            accepted_pos[bin][pos]+=1
    all_pairs[bin]+=1
    all_pairs_pos[bin][pos]+=1
for bin in accepted:
    print(bin,':',round(100*float(accepted[bin]/all_pairs[bin]),2),'% accepted',accepted[bin])
    for pos in accepted_pos[bin]:
        #print(bin,pos,':',float(accepted_pos[bin][pos]/all_pairs_pos[bin][pos]))
        pass
print(final_pair_file,len(correct_pairs),'out of',len(lines))
with open(os.path.join(final_pair_file),'w') as buf:
    buf.write('\n'.join(correct_pairs)+'\n')