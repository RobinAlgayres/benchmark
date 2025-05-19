import os,sys,json
import numpy as np
from scipy import stats

def get_score(res_dir,tag):
    fid_to_remove=['gpt2','roberta-base']
    lastbin,high,low,freqs=[],[],[],[]
    fids=[]
    for fid in os.listdir(res_dir):
        if fid in fid_to_remove:
            continue
        fids.append(fid)
    fids.sort()
    for fid in fids:
        if 'json' in fid:
            with open(os.path.join(res_dir,fid)) as buf:
                res=json.load(buf)
            if 'wordswap' in res_dir:
                freqs.append(res['AVG_BIN'][1:])
            else:
                freqs.append(res['AVG_BIN'])
            lastbin.append(res['AVG_BIN'][-1])
        else:
            with open(os.path.join(res_dir,fid)) as buf:
                res=buf.readlines()[0].split(' ')
            freqs.append([float(s) for s in res[1:-1]])
            lastbin.append(float(res[-1]))
    freqs=np.array(freqs)
    lastbin=np.around(lastbin,3)
    avg_per_freqs=np.mean(freqs,axis=0)
    high_minus_low=avg_per_freqs[-1]-avg_per_freqs[0]
    #print('high-minus-low',tag,np.around(high_minus_low,3))
    
    return lastbin,freqs,fids
    
def get_spread(table,tag):
    spread=(max(table[:,0])-min(table[:,0]))/(max(table[:,-1])-min(table[:,-1]))
    spread=round(spread,2)

    #print(tag,'spread')
    return spread

def get_all_avg(syn,infl,word,fids,tag):
    all_avg_0=np.around((syn[:,0]+infl[:,0])/2,2).reshape(-1,1)
    all_avg=np.around((syn[:,1:]+infl[:,1:]+word)/3,2)
    all_avg=np.concatenate((all_avg_0,all_avg),axis=1)
    for i in range(len(fids)):
        print(fids[i],' '.join([str(e) for e in all_avg[i]]))
    avg_per_freqs=np.mean(all_avg,axis=0)
    high_minus_low=avg_per_freqs[-1]-avg_per_freqs[0]
    print('high-minus-low',tag,np.around(high_minus_low,3))
    return all_avg

def sort_along_bis(first,second):
    indices=np.argsort(first)
    first=first[indices]
    second=second[indices]

    pr=stats.pearsonr(first,second)
    sr=stats.spearmanr(first,second)
    print('pearson:',np.around([pr.statistic,pr.pvalue],2))
    print('spearman:',np.around([sr.statistic,sr.pvalue],2))

def sort_along_freq(source,tag):
    indices=np.arange(len(source[0]))
    if len(indices)==11:
        frequencies=np.array([0,1,2,4,8,16,32,64,128,256,512])
        offset=1
    else:
        frequencies=np.array([1,2,4,8,16,32,64,128,256,512])
        offset=0

    corr_bin=[]
    high_bin=source[:,-1]
    corr_bin=[]
    low_bin=np.mean(source[:,:1+offset],axis=1)
    res=stats.spearmanr(high_bin,low_bin)
    srl,svl=res.statistic,res.pvalue
    
    mid_bin=source[:,-2]
    res=stats.spearmanr(high_bin,mid_bin)
    srh,svh=res.statistic,res.pvalue
    #corr_bin='/'.join([str(round(srl,2)),str(round(srh,2))])
    corr_bin='/'.join([str(round(srl,2)),str(round(svl,2)),str(round(srh,2)),str(round(svh,2))])
    
    corr=[]
    for model in source:
        #res=stats.pearsonr(model,frequencies)
        #pr,pv=res.statistic,res.pvalue
        #print('pearson',np.around([pr*pr,pv],2),model,frequencies)
        
        #for model in source:
        res=stats.spearmanr(model,indices)
        sr,sv=res.statistic,res.pvalue
        corr.append(sr)
        

    if True:
        zs = [fisher_z(r) for r in corr]
        z_avg = np.mean(zs)
        r_avg = inverse_fisher_z(z_avg)
    else:
        r_avg = np.mean(corr)
    #print(tag,f"avg corr: {r_avg:.3f}")
    return round(r_avg,2),corr_bin

def fisher_z(r):
        # Step 1: Convert to Fisher Z
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)    

def sort_along(source,freqs):
    indices=np.argsort(source)
    source=source[indices]
    freqs=freqs[indices]
    high=np.mean(freqs[:,-3:],axis=1)
    low=np.mean(freqs[:,:3],axis=1)
    prh,prl=stats.pearsonr(source,high),stats.pearsonr(source,low)
    srh,srl=stats.spearmanr(source,high),stats.spearmanr(source,low)
    print('pearson high:',np.around([prh.statistic,prh.pvalue],2),'low:',np.around([prl.statistic,prl.pvalue],2))
    print('spearman high',np.around([srh.statistic,srh.pvalue],2),'low:',np.around([srl.statistic,srl.pvalue],2))

if __name__=='__main__': 
    #result_dirs=['results_100M_withdet','results_10M_withdet']
    result_dirs=['results_10M_matrix','results_100M_matrix']
    for result_dir in result_dirs:
        print(result_dir)
        if '_matrix' in result_dir:
            blimp_dir=os.path.join('blimp/','_'.join(result_dir.split('_')[:-1]))
        else:
            blimp_dir=os.path.join('blimp/',result_dir)
        wordswap_dir=os.path.join(result_dir,'wordswap')
        syntax_dir=os.path.join(result_dir,'syntax')
        inflswap_dir=os.path.join(result_dir,'inflswap')
       
        blimp,blimp_per_freq,fids=get_score(blimp_dir,'BLIMP')
        syn,syn_per_freq,_=get_score(syntax_dir,'SYNTAX')
        infl,infl_per_freq,_=get_score(inflswap_dir,'INFLECTION')
        word,word_per_freq,_=get_score(wordswap_dir,'WORD')
        all_per_freq=get_all_avg(syn_per_freq,infl_per_freq,word_per_freq,fids,'ALL')

        blimp_s=get_spread(blimp_per_freq,'BLIMP')
        syn_s=get_spread(syn_per_freq,'SYNTAX')
        infl_s=get_spread(infl_per_freq,'INFLECTION')
        word_s=get_spread(word_per_freq,'WORD')
        all_s=get_spread(all_per_freq,'ALL')
        print('Spread:',blimp_s,all_s,word_s,infl_s,syn_s)
        
        syn_r,syn_r_bin=sort_along_freq(syn_per_freq,'SYNTAX')
        infl_r,infl_r_bin=sort_along_freq(infl_per_freq,'INFLECTION')
        word_r,word_r_bin=sort_along_freq(word_per_freq,'WORD')
        all_r,all_r_bin=sort_along_freq(all_per_freq,'ALL')
        blimp_r,blimp_r_bin=sort_along_freq(blimp_per_freq,'BLIMP')
        #print('CORR_BIN:',blimp_r_bin,all_r_bin,word_r_bin,infl_r_bin,syn_r_bin)
        print('CORR:',blimp_r,all_r,word_r,infl_r,syn_r)
        
        #print('BLIMP/WORD')
        #sort_along(blimp,word_per_freq)
        #print('BLIMP/INFL')
        #sort_along(blimp,infl_per_freq)
        #print('BLIMP/SYN')
        #sort_along(blimp,syn_per_freq)
        