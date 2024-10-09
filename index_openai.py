import os
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from uuid import uuid4
from transformers import BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma

app = FastAPI()
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Configuration
PDF_PATH = os.getenv("PDF_PATH", "./data/falcon-users-guide-2021-09-compressed.pdf")
PDF_PATHS = os.getenv("PDF_PATHS", "./data")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_langchain_db")

# Define request and response models
class PromptRequest(BaseModel):
    prompt: str

class ResponseModel(BaseModel):
    prompt: str
    context: str
    response: str

def extract_model_response(full_response):
    """Extracts the model's response from the full response text."""
    delimiter = "Please provide a helpful response to the user. "
    start_index = full_response.find(delimiter)
    if start_index != -1:
        return full_response[start_index + len(delimiter):].strip()
    return "No valid response found."

@app.on_event("startup")
async def startup_event():
    """Initializes embeddings and vector store on startup."""
    global openai_embedding, vector_store

    openai_embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=768)
    documents = load_documents(PDF_PATHS)

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=openai_embedding,
        persist_directory=CHROMA_DB_DIR,
    )

    vector_store.add_documents(
        documents=documents, ids=[str(uuid4()) for _ in range(len(documents))]
    )

def load_documents(directory):
    """Loads documents from the specified directory."""
    documents = []
    for file in os.listdir(directory):
        loader = PyPDFLoader(os.path.join(directory, file))
        document = loader.load()
        documents.append(Document(page_content=document[0].page_content, id=str(uuid4())))
    return documents

@app.post("/upload_document")
async def upload_document(file: bytes = Form(...)):
    """Endpoint to upload a new document."""
    try:
        loader = PyPDFLoader(file)
        document = loader.load()
        doc = Document(page_content=document[0].page_content, id=str(uuid4()))
        vector_store.add_documents(documents=[doc], ids=[doc.id])

        return {"message": "Document uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(prompt: str = Form(...)):
    """Generates a response based on the provided prompt."""
    try:
        results = vector_store.similarity_search(prompt, k=10)
        context = results[0].page_content if results else "No relevant context found."

        rag_prompt = (
            f"You are a helpful engineer who is trying to help a user with a problem. "
            f"The user has a problem statement: {prompt}. The user has provided the following context: {context}. "
            "Please provide a helpful response to the user."
        )

        llm = OpenAI()
        output = llm.invoke(rag_prompt)

        return {
            "prompt": prompt,
            "context": context,
            "response": extract_model_response(output),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)