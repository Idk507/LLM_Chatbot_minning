import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

embedding_model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name = embedding_model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

documents = DirectoryLoader('./', glob="*.pdf", loader_cls=PyPDFLoader).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
metadata_config = {"hnsw:space": "cosine"}
persist_dir = "vectorbase/mining"
vector_store = Chroma.from_documents(texts,embeddings,persist_directory = persist_dir, collection_metadata = metadata_config)

