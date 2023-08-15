# -*- coding: utf-8 -*-
"""
Created on Tue May 23 03:54:57 2023

@author: mogit
"""
#final draft consists of all codes
import re
import string
import unidecode
import openai
from PyPDF2 import PdfReader
from keybert import KeyBERT
import nltk
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelWithLMHead 


def clean_words(text):
    """Basic cleaning of texts"""
    
    # remove html
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()

    #remove accented characters
    text=unidecode.unidecode(text) #we have accented characters like a^ etc, so to remove that we are performing 
    
    #to make words into lowercase
    text=text.lower()
    
    # also we are not performing stemming and lemmatization since it will change the context of passage and other words in chapter text
    return text

def extract_text_from_PDF(filepath):
    
    with open (filepath,'rb') as file:
        para=[]
        pdf_reader=PdfReader(file)
        
        for page in pdf_reader.pages:
            text=page.extract_text()
            text=clean_words(text)
            para.append(text.split('\n\n') )  
        return para

def para_summarizer_t5(text):
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    input_encode=tokenizer.encode("summarize: " +text, return_tensors='pt')
    summary = model.generate(input_encode, max_length=200)

    final_summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
    return(final_summary_text)
    
def keyword_noun_extractor(text):
    nouns=[]
    for word,pos in nltk.pos_tag(nltk.word_tokenize(str(text))):
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                 nouns.append(word)
                 
    text=" ".join(nouns)          
    keyword_model=KeyBERT(model='all-mpnet-base-v2')
    keywords=keyword_model.extract_keywords(text,stop_words='english',
                                            highlight=False,top_n=2,
                                            keyphrase_ngram_range=(1,1 ))
    return [i for i,j in keywords]

def get_question(answer, context, max_length=80):
  tokenizer=AutoTokenizer.from_pretrained('T5-base')
  model=AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-question-generation-ap', return_dict=True)
  input_text = "answer: %s  context: %s " % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               no_repeat_ngram_size=2,
               max_length=max_length)
  ques=tokenizer.decode(output[0],skip_special_tokens=True)

  return (ques,answer)

def distractor_using_openAI(word):
    openai.api_key = "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"
    
    prompt_task=f"generate 1 distractor for the word ''{word}'"
    responses = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt_task,
                                        temperature=0.5, 
                                        max_tokens=50,
                                        n=1)
    return [(response['text']) for response in responses.choices]
        

def integrated_function(para):
        #dist_words=[]
        summarized=para_summarizer_t5(para)
        keyword=keyword_noun_extractor(summarized)
        question_ans=get_question(keyword, summarized)
        
        #for words in keyword:
            #dist_words.append(distractor_using_openAI(keyword))
            
        return (question_ans)
        
    
#Main program
if __name__=="__main__":
    reader =["C:/Users/mogit/Akaike assignments/NLP assignment/Datasets/chapter-2.pdf"]
    
    ques_ans=[]
    for url in reader:
        text_lists=(extract_text_from_PDF(url))
        for para in text_lists:
            ques=integrated_function(str(para))
            ques_ans.append(ques)
            print(ques)
            
            
    with open('mcq_questions_t5.txt','w')as file:
       file.write('\n'.join(f'{ques[0]} {ques[1]}' for ques in ques_ans))
        
    
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    