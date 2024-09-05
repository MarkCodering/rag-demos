# pip install transformers
# pip install bitsandbytes
# pip install -qU langchain_community pypdf
# pip install -qU "langchain-chroma>=0.1.2"
# pip install -U sentence-transformers
# pip install sentence_transformers
# pip install langchain==0.0.174 -i https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com

from uuid import uuid4

from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

prompt = input("Enter a prompt: ")

embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m", model_kwargs = {'device': 'cuda'}, show_progress=True)

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

pipe = pipeline(model="meta-llama/Meta-Llama-3-8B", model_kwargs={"load_in_4bit": True}, device_map="auto")
output = pipe(rag_prompt, do_sample=True, top_p=0.95)

print("\n" + output[0]["generated_text"] + "\n")