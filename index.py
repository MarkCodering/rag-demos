import torch
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from uuid import uuid4
from transformers import pipeline, BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

app = FastAPI()
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)  # You can also try load_in_4bit


# Define request and response models
class PromptRequest(BaseModel):
    prompt: str


class ResponseModel(BaseModel):
    prompt: str
    context: str
    response: str


def extract_model_response(full_response):
    # The key part of the full response that we want starts after this phrase
    delimiter = "Please provide a helpful response to the user. "

    # Find the index of the delimiter
    start_index = full_response.find(delimiter)

    if start_index != -1:
        # The actual model response starts after the delimiter
        # Add length of the delimiter to start_index to get the actual response
        model_response = full_response[start_index + len(delimiter) :].strip()
        return model_response
    else:
        return "No valid response found."


# Load model and set up necessary components when the app starts
@app.on_event("startup")
async def startup_event():
    global embeddings, vector_store, pipe, document_1

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
        show_progress=True,
    )

    # Load the document (replace with the correct path to your PDF)
    loader = PyPDFLoader("./data/falcon-users-guide-2021-09-compressed.pdf")
    document = loader.load()

    document_1 = Document(
        page_content=document[0].page_content,
        id=1,
    )

    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    # Add the document to the vector store
    documents = [document_1]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"load_in_4bit": True},
    )


# Define the POST endpoint
@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(prompt: str = Form(...)):
    try:
        # Perform similarity search
        results = vector_store.similarity_search(prompt, k=5)
        context = results[0].page_content if results else "No relevant context found."

        # Generate the response using the LLM
        rag_prompt = (
            f"You are a helpful engineer who is trying to help a user with a problem. "
            f"The user has a problem statement: {prompt}. The user has provided the following context: {context}. "
            "Please provide a helpful response to the user."
        )

        response = pipe(str(rag_prompt), max_new_tokens=512)
        output = response[0]["generated_text"]
        print(extract_model_response(output))

        return {
            "prompt": prompt,
            "context": context,
            "response": extract_model_response(output),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application if executed as a script
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
