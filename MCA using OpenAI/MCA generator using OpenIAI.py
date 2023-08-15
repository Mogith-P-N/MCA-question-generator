# -*- coding: utf-8 -*-
"""
Created on Mon May 22 01:31:57 2023
@author: mogit
"""
import openai
import re
import unidecode
from PyPDF2 import PdfReader

openai.api_key = "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"


def get_mca_questions(text):
    prompt=f"create a abstractive summary of given {text} and generate a multiple choice question with from summary with 4 options where question should reflect reader's understanding of comprehension and answers should be in one word with 2 correct answers in options and distractors in other options compulsorily. label only right answers as '(correct)'. display only MCQ and options"
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,temperature=0.9, 
                                        max_tokens=50, 
                                        top_p=1, 
                                        frequency_penalty=1, 
                                        presence_penalty=1,)
    
    return((response['choices'][0]['text']))

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
    
    # also we are not performing stemming and lemmatization since it will change the context of passage and keywords in text
    return text

def extract_text_from_PDF(filepath):
    para=[]
    
    with open (filepath,'rb') as file:
        
        pdf_reader=PdfReader(file)
        
        for page in pdf_reader.pages:
            text=page.extract_text()
            text=clean_words(text)
            para.append(text.split('\n\n'))
            
        return para

#Main program
if __name__=="__main__":
    
    chapters =["C:/Users/mogit/Akaike assignments/NLP assignment/Datasets/chapter-2.pdf",
               "C:/Users/mogit/Akaike assignments/NLP assignment/Datasets/chapter-3.pdf",
               "C:/Users/mogit/Akaike assignments/NLP assignment/Datasets/chapter-4.pdf" ]
    
    Mcq_questions=[]
    for url in chapters:
        paragraph=extract_text_from_PDF(url)
        for para in paragraph:
            ques=get_mca_questions(para)
            print(ques)
            Mcq_questions.append(ques)

    with open('mcq_questions.txt','w')as file:
        file.write('\n'.join(Mcq_questions))

    

    











