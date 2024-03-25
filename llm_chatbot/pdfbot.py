from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

def ask_pdf_question(pdf_path, user_question):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    
    embeddings = HuggingFaceEmbeddings()

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    docs = knowledge_base.similarity_search(user_question)
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 64})
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=user_question)

    return response


pdf_path = "path/to/your/pdf/file.pdf"
user_question = input("Ask Question about your PDF: ")
result = ask_pdf_question(pdf_path, user_question)
print(result)
