from langdetect import detect
from googletrans import Translator
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFaceHub
from langdetect import detect
from googletrans import Translator

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

'''def detect_language(user_input):
    try:
        lang = detect(user_input)
        return lang
    except:
        return 'en'  # Default to English if language detection fails

# Function to translate text
'''
def translate_text(text, target_language):
    translator = Translator()

    try:
        translated_text = translator.translate(text, dest=target_language).text
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

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
# Function to display language options and get user input
def choose_language():
    print("Choose your preferred language:")
    print("1. Hindi")
    print("2. Bengali")
    print("3. Telugu")
    print("4. Marathi")
    print("5. Tamil")
    print("6. Urdu")
    print("7. Gujarati")
    print("8. Malayalam")
    print("9. Kannada")
    print("10. Oriya")
    print("11. Punjabi")
    print("12. Assamese")
    print("13. Maithili")
    print("14. Santali")
    print("15. Nepali")
    
    language_choice = input("Enter the number corresponding to your choice: ")
    
    
    language_mapping = {
        '1': 'hi',  # Hindi
        '2': 'bn',  # Bengali
        '3': 'te',  # Telugu
        '4': 'mr',  # Marathi
        '5': 'ta',  # Tamil
        '6': 'ur',  # Urdu
        '7': 'gu',  # Gujarati
        '8': 'ml',  # Malayalam
        '9': 'kn',  # Kannada
        '10': 'or',  # Oriya
        '11': 'pa',  # Punjabi
        '12': 'as',  # Assamese
        '13': 'mai',  # Maithili
        '14': 'sat',  # Santali
        '15': 'ne',  # Nepali
    }
    
    return language_mapping.get(language_choice, 'en')  # Default to English if choice is not recognized
import re

def clean_text(text):
    cleaned_text = re.sub(r'\n', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

while True:
    inp = input("\n Enter query : ")
    response = qa(inp)
    #print(response)
    result_text = response.get('result', '')
    cleaned_result = clean_text(result_text)
    print(cleaned_result)
    target_language = choose_language()
    translated_response = translate_text(cleaned_result, target_language)
    
    print(f"\nTranslated Response ({target_language}): {translated_response}")

