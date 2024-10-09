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
    delimiter = "Please provide a helpful response to the user. "
    start_index = full_response.find(delimiter)
    if start_index != -1:
        return full_response[start_index + len(delimiter) :].strip()
    return "No valid response found."


@app.on_event("startup")
async def startup_event():
    global openai_embedding, vector_store

    openai_embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=768)

    documents = []

    for file in os.listdir(PDF_PATHS):
        loader = PyPDFLoader(f"{PDF_PATHS}/{file}")
        document = loader.load()
        documents.append(
            Document(page_content=document[0].page_content, id=str(uuid4()))
        )

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=openai_embedding,
        persist_directory=CHROMA_DB_DIR,
    )

    vector_store.add_documents(
        documents=documents, ids=[str(uuid4()) for _ in range(len(documents))]
    )


@app.post("/upload_document")
async def upload_document(file: bytes = Form(...)):
    try:
        loader = PyPDFLoader(file)
        document = loader.load()

        document = Document(page_content=document[0].page_content, id=1)
        vector_store.add_documents(documents=[document], ids=[str(uuid4())])

        return {"message": "Document uploaded successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(prompt: str = Form(...)):
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
            "response": output,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
