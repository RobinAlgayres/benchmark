import os,sys
from spellchecker import SpellChecker


# compute inflection of words without constraint on the number of letters

def jj_inflections(adj):
    if adj[-1]=='e':
        comparative = adj[:-1] + 'er' 
        superlative = adj[:-1] + 'est'  
    elif adj[-1]=='y':
        comparative = adj[:-1] + 'ier'  
        superlative = adj[:-1] + 'iest' 
    elif len(adj) == 3 and adj[-1] not in 'aeiou':
        #double letter for three letter adj
        comparative = adj + adj[-1] + 'er'
        superlative = adj + adj[-1] + 'est'
    else:
        comparative = adj + 'er'  
        superlative = adj + 'est'

    output=[]
    for word in [adj,comparative,superlative]:
        output.append(word)
        output.append('un'+word) 
    # Return all the inflections
    return output

def nn_inflections(noun):
  # Third-person singular present: add "s" or "es"
    if noun[-1] in ['s', 'x', 'z'] or noun[-2:] in ['sh', 'ch']:
        third_person = noun + 'es'  # For nouns ending in 's', 'sh', 'ch', 'x', or 'z' (e.g., "pass" → "passes")
    elif noun[-1]=='y':
        third_person = noun[:-1] + 'ies'
    else:
        third_person = noun + 's'  # Regular case (e.g., "run" → "runs")

    output=[]
    for word in [noun,third_person]:
        output.append(word)
        output.append('un'+word) 
    return output

def vb_inflections(verb,spell):
    # Past tense: regular verbs add "ed"
    if verb[-1]=='e':
        past_tense = verb + 'd'  # For verbs ending in 'e' (e.g., "love" → "loved")
        present_participle = verb[:-1] + 'ing'  # Remove 'e' for verbs ending in 'e' (e.g., "love" → "loving")
    elif verb[-1]=='y':
        past_tense = verb[:-1] + 'ied' #study studied
        if len(spell.known([past_tense]))==0: 
            past_tense=verb + 'ed'#play played
        present_participle = verb + 'ing' #study studying
    elif len(verb) == 3 and verb[-1] not in 'aeiou':
        #double letter for three letter adj
        past_tense = verb + verb[-1] + 'ed'
        present_participle = verb + verb[-1] + 'ing'
    else:
        past_tense = verb + 'ed'  # Regular case for past tense (e.g., "talk" → "talked")
        present_participle = verb + 'ing'  # Regular case (e.g., "talk" → "talking")
    
    # Third-person singular present: add "s" or "es"
    if verb[-1] in ['s', 'x', 'z'] or verb[-2:] in ['sh', 'ch']:
        third_person = verb + 'es'  # For verbs ending in 's', 'sh', 'ch', 'x', or 'z' (e.g., "pass" → "passes")
    elif verb[-1]=='y':
        third_person = verb[:-1] + 'ies'
    else:
        third_person = verb + 's'  # Regular case (e.g., "run" → "runs")

    output=[]    
    for word in [verb,past_tense,present_participle,third_person]:
        output.append(word)
        output.append('un'+word) 

    return output

def deal_with_last_letter(word):
    if len(word)<3:
        return word
    if word[-1]==word[-2]: #run running, bed bedded 
        word=word[:-1]
    elif word[-1]=='i': #study studied
        word=word[:-1]+'y'
    elif word[-1] not in 'aeiouy': #love loving, note noted
        word=word[:-1]+'e'
    return word

def get_base_form(init_word,pos,spell):
    word=init_word

    #remove the common suffix -un
    if word[:2]=='un':
        word=word[2:]    
        if len(spell.known([word]))==0:
            #if the word has been changed into a non-word
            word=init_word    

    #remove the common prefix (-ed,-s,-ing,-er,-est)
    if word[-3:]=='ing' and pos=='VERB':
        word=word[:-3]
    elif word[-2:]=='ed' and pos=='VERB':
        word=word[:-2] 
    elif word[-2:]=='er' and pos=='ADJ':
        word=word[:-2]
    elif word[-3:]=='est' and pos=='ADJ':
        word=word[:-3]
    elif word[-1:]=='s' and pos in ['VERB','NOUN'] and len(word)>3: #remove plural
        word=word[:-1]
        if word[-2:] in ['se', 'xe', 'ze'] or word[-3:] in ['she', 'che']:
            word=word[:-1]
        elif word[-2:]=='ie':
            word=word[:-2]+'y'
            

    if len(spell.known([word]))==0:
        word=deal_with_last_letter(word)
        if len(spell.known([word]))==0:
            #if the modified word does not belong to dictionnary anymore
            return init_word
    return word


def format_word(form,pos):
    spell = SpellChecker()
    assert pos in ['VERB','NOUN','ADJ']
    form=get_base_form(form,pos,spell)
    forms=[form]
    # no constraint on the number of letters
    if pos=='VERB':
        forms=vb_inflections(form,spell)
    elif pos=='ADJ':
        forms=jj_inflections(form)
    elif pos=='NOUN':
        forms=nn_inflections(form)
    
    final_forms=[]
    for form in forms:
        if len(spell.known([form]))>0 and len(form)>2:
            final_forms.append(form) 
    return final_forms

if __name__=='__main__':
    words =[('happier','ADJ'),('happy','ADJ'),('unhappier','ADJ'),('running','VERB'),('faster','ADJ'),('trees','NOUN'),('happiness','NOUN')]
    #words=[('stable','ADJ')]
    for word in words:
        form,pos=word
        forms=format_word(form,pos)
        print(form,':',forms)


