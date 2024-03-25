import speech_recognition as sr 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import os
import playsound
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFaceHub
import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()

engine = pyttsx3.init()


def listen_query():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        
    try:
        text = r.recognize_google(audio)
        print("You said:", text) 
        return text
    
    except:
        print("Speech recognition failed")
        return ""
        
def speak_response(text):
    print("Assistant:", text)   
    engine.say(text)
    engine.runAndWait()
    

# HuggingFace Setup
model_name = "microsoft/DialoGPT-medium" 
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(text_query):

    inputs = tokenizer([text_query], return_tensors="pt")
    reply_ids = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.batch_decode(reply_ids)[0]

    return response

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


llm_name = "HuggingFaceH4/zephyr-7b-beta"

config = {
'max_new_tokens': 1024,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

llm = HuggingFaceHub(
    repo_id=llm_name, model_kwargs={"temperature": 0.5, "max_length": 64}
)

print("Chatbot initiated")


prompt_template = """Use the following pieces of information to answer the user's question.
Act like a mining chatbot and provide only honest answers.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the meaningful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="vectorbase/mining", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})

chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents = True,
    chain_type_kwargs= chain_type_kwargs,
    verbose=True
)
import re
def clean_text(text):
    cleaned_text = re.sub(r'\n', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

print("Chatbot is ready...")

while True:
    query = listen_query()
    if query:
        response = generate_response(query)
        response = qa(response) 
        result_text = response.get('result', '')
        cleaned_result = clean_text(result_text)
        print(cleaned_result)
        speak_response(cleaned_result)
        #print(response)