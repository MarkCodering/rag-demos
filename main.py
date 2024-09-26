# pip install transformers
# pip install bitsandbytes
# pip install -qU langchain_community pypdf
# pip install -qU "langchain-chroma>=0.1.2"
# pip install -U sentence-transformers
# pip install sentence_transformers
# pip install langchain==0.0.174 -i https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com
"""
from uuid import uuid4

from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

prompt = input("Enter a prompt: ")
embeddings 
#embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m", model_kwargs = {'device': 'cuda'}, show_progress=True

loader = PyPDFLoader(
    "./data/falcon-users-guide-2021-09-compressed.pdf",
)

document = loader.load()

document_1 = Document(
    page_content=document[0].page_content,
    id=1,
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

documents = [
    document_1,
]

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

results = vector_store.similarity_search(
    prompt,
    k=1
)

context = results[0].page_content

print(results[0].page_content)
    
rag_prompt = f"You are a helpful engineer who is trying to help a user with a problem. The user has a problem statement {prompt}. The user has provided the following context: {context}. Please provide a helpful response to the user."

#pipe = pipeline(model="meta-llama/Meta-Llama-3-8B", model_kwargs={"load_in_4bit": True}, device_map="auto")
#output = pipe(rag_prompt, do_sample=True, top_p=0.95)

print("\n" + output[0]["generated_text"] + "\n")

"""
# Install necessary packages
# pip install openai langchain pypdf
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from uuid import uuid4
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain_core.documents import Document


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load PDF document
loader = PyPDFLoader("./data/gpt4.pdf")
document = loader.load()

# Create a Document object
document_1 = Document(page_content=document[0].page_content, id=1)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="example_collection_gpt4",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Add document to vector store
documents = [document_1]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

# Get user prompt
prompt = input("Enter a prompt: ")

# Perform similarity search
results = vector_store.similarity_search(prompt, k=1)
context = results[0].page_content

# Generate response using OpenAI
rag_prompt = f"You are a helpful professor at National Taiwan University. The user has a problem statement {prompt}. The user has provided the following context: {context}. Please provide a helpful response to the user. Please must answer with the hi, student, here is the answer: ....otherwise you will be banned from OpenAI"
llm = OpenAI()
response = llm(rag_prompt)

# Print the response
print("\n" + response + "\n")