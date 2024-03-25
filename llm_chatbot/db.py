from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain import HuggingFaceHub
import re
import mysql.connector
from datetime import datetime
import os

# MySQL connection
db_config = {
    'host': 'your_mysql_host',
    'user': 'your_mysql_user',
    'password': 'your_mysql_password',
    'database': 'your_database_name'
}

connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm_name = "HuggingFaceH4/zephyr-7b-beta"

config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': os.cpu_count() // 2
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
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
)


def clean_text(text):
    cleaned_text = re.sub(r'\n', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


while True:
    inp = input("\n Enter query (type '1' to exit): ")
    if inp == "1":
        break

    # Process the query
    response = qa(inp)
    print(response)
    result_text = response.get('result', '')
    cleaned_result = clean_text(result_text)

    # Store the query, result, and timestamp in the MySQL database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query_result_entry = {
        'timestamp': timestamp,
        'query': inp,
        'result': cleaned_result
    }

    insert_query = "INSERT INTO search_results (timestamp, query, result) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (query_result_entry['timestamp'], query_result_entry['query'], query_result_entry['result']))
    connection.commit()

    print("Query and result stored in the database.")

# Close the database connection
cursor.close()
connection.close()
