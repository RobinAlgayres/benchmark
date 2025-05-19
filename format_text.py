import numpy as np
import os,sys
import re

def space_characters(word,map_letters):
    #space symbols from letters and compute the symbol ratio
    r=0
    new_word=''
    has_foreign_letters=False
    lower_word=word.lower()
    for i in range(len(word)):
        lower_char=lower_word[i]
        char=word[i]

        if lower_char in map_letters['foreigns']:
            has_foreign_letters=True
        
        if lower_char not in map_letters['letters']: 
            #splitting symbols from words
            if char!='-':
                new_word+=' '+char+' '
            else:
                try:
                    #if it is a - in between numbers, the - is spaced out
                    if word[i-1] not in map_letters['letters'] and word[i+1] not in map_letters['letters']:
                        new_word+=' '+char+' '
                    else:
                        #in this case, the - is joining word together
                        new_word+=char
                except:
                    #in this case, the - is joining word together
                    new_word+=char
            if lower_char not in '!"$%&\'()*,-.0123456789:;?@[]':
                #it is a real symbol
                r+=1
        else:
            new_word+=char

    #remove beginning and trailing char
    new_word=' '.join(list(filter(None,new_word.split(' '))))
    new_word=new_word.strip()
    ratio_symbols=float(r)/float(len(word))
    return new_word,has_foreign_letters,ratio_symbols

def format_words(sentence,map_letters):
    #rejecting full sentences if does not pass quality filters

    new_sentence=[]
    foreign_word,word_with_symbols,skipped_words=0,0,0
    if sentence[0][0]=='[' and sentence[-1][-1]==']':
        #a lot of sentence are useless contextual information
        return None,'useless_context_information'

    for word in sentence:
        assert len(word)>0,sentence
        #rejecting the full sentence
        if 'javascript' in word or 'html' in word:
            return None,'web'
        if 'http' in word or 'www' in word:
            return None,'web'
        # skipping word with useless contextual information
        if '[' in word and ']' in word:
            continue
        
        word,has_foreign_letters,ratio_symbols=space_characters(word,map_letters)


        if ratio_symbols>0:
            word_with_symbols+=1

        if '|' in word: #I need that symbol for other things later
            skipped_words+=1
            continue
        
        if '-' not in word and len(word)>40:
            #if it is not a compound word and the word is too long (probably URL)
            skipped_words+=1
            continue
        
        if len(word)>20 and ratio_symbols>0.3:
            #probably a non-word
            skipped_words+=1
            continue

        #some sentences are only composed of one symbol, like *
        if len(word)==1 and ratio_symbols==1:
            skipped_words+=1
            continue

        if has_foreign_letters:
            foreign_word+=1
       
        new_sentence.append(word)

    ratio_skipped_words=float(skipped_words)/float(len(sentence))
    ratio_foreign_words=float(foreign_word)/float(len(sentence))
    ratio_words_with_symbols=float(word_with_symbols)/float(len(sentence))
    if ratio_foreign_words>0.2:
        #probably not an english sentence
        return None,'too_many_foreign_words'
    if ratio_words_with_symbols>0.3:
        #probably not a real sentence
        return None,'too_many_words_with_symbols'
    if ratio_skipped_words>0.2:
        #noisy sentence
        return None,'too_many_skipped_words'
    if len(new_sentence)==0:
        return None,'empty_lines'

    return new_sentence,'accepted'

if __name__=='__main__':
    #1- removing sentences that contains headers, 
    # too many foreign words, URL, very long words, words with too many symbols
    #2- addings speaker tags in a consistant way across different dialogue datasets
    #3- numbers are split into individual figures
    #4- allowed symbols are spaced out from words: !"$%&\'()*,-.0123456789:;?@[] 
    #5- only '-' are kept attached to words for LMs to identify compound words

    #path to BabyLM dataset
    data='../shared/data/BabyLM_2024/text_data/train_10M/'
    #path to output dataset, will be created
    output_char_dir='BabyLM_2024_formatted_/train_10M/'

  

    if not os.path.isdir(output_char_dir):
        os.makedirs(output_char_dir)
    
    map_letters={}
    map_letters['letters']='abcdefghijklmnopqrstuvwxyz'
    map_letters['foreigns']='àáâãäåæçèéêëìíîïðñòóôõöøùúûüýÿāăąćĉčďđēĕėęěĝğġģĥħĩīĭįıĵķĺļľłńņňŋōŏőœŕŗřśŝşšţťũūŭůűŵŷźżžơưǎǐǒǔǚǧǩǫǵǹȁȃșțḋḍḏḗḡḥḩḫḱḵḷḻḿṁṃṅṇṉṓṗṙṛṟṣṫṭṯṳṹẏẓạảấầẩậắằẵặẹẻẽếềểễệỉịọỏốồổỗộớờởợụủứửữự'


    all_words=0
    for fid in os.listdir(data):
        path=os.path.join(data,fid)
        output_char_file=os.path.join(output_char_dir,fid)
        formatted_chars=[]
        print(fid)
        if os.path.isfile(output_char_file):
            continue
        c=0
        skipped={}
        with open(path) as buf:
            for line in buf:
                chars=line.strip()
                chars=chars.replace('...','.')
                chars=chars.replace('\t',' ')
                sentence=list(filter(None,chars.split(' ')))

                if len(sentence)==0:
                    continue
                nb_words=len(sentence)
                sentence,skipping=format_words(sentence,map_letters)
                if sentence is None:
                    if skipping not in skipped:
                        skipped[skipping]=0
                    skipped[skipping]+=1
                    continue
                all_words+=nb_words
                if fid in ['childes.train','switchboard.train']:
                    if sentence[0] in ['a :','b :']:
                        spk_id=sentence[0]
                    elif sentence[0][0]=='*':
                        #will typically be: '* CHI :'
                        spk_id=sentence[0].split(' ')[1]
                        #if spk_id not in ['chi','mot','fat', 'inv' ,'exp' ,'gma' ,'par']:
                        #    spk_id='unk_ch'
                    else:
                        #a lot of "[not clear what this means]"
                        continue
                    sentence='<spk_id> '+spk_id+' <s_turn> '+' '.join(sentence[1:])+' <e_turn>'
                elif fid in ['open_subtitles.train']:
                    if sentence[0]=='-':
                        #open subtitles somethimes starts with a '-'
                        sentence=sentence[:-1]
                    elif sentence[0][-1]==':':
                        #open subtitles sometimes use speaker id
                        continue
                    sentence='<spk_id> unk_os <s_turn> '+' '.join(sentence)+' <e_turn>'
                elif fid=='simple_wiki.train':
                    if sentence[0]=='=':
                        #title line
                        continue
                    if len(sentence)<=4:
                        #paragraphe names
                        continue
                    sentence=' '.join(sentence)
                elif fid=='gutenberg.train':
                    if sentence[0][0]=='*':
                        #chapter titles
                        continue
                    sentence=' '.join(sentence)
                elif fid=='bnc_spoken.train':
                    sentence=' '.join(sentence)

                sentence=fid+'|'+sentence
                formatted_chars.append(sentence)
        with open(output_char_file,'w') as buf:
            buf.write('\n'.join(formatted_chars)+'\n')
        print('total word accepted so far:',all_words)
        print('number sentences removed:',skipped)
    print('Final all kept words:',all_words)
