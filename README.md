# README

## Overview

This repository provides a framework for leveraging advanced natural language processing (NLP) techniques to perform document similarity searches and generate context-aware responses. The project utilizes several state-of-the-art libraries and models, including Hugging Face Transformers, Langchain, and Chroma, to process and analyze PDF documents.

## Features

- **PDF Document Loading**: Load and process PDF documents using `PyPDFLoader`.
- **Embeddings Generation**: Generate embeddings using Hugging Face's `HuggingFaceEmbeddings`.
- **Vector Store Management**: Store and manage document embeddings with `Chroma`.
- **Similarity Search**: Perform similarity searches to find relevant document content based on user prompts.
- **Contextual Response Generation**: Generate context-aware responses using a pre-trained language model pipeline.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- CUDA-compatible GPU for optimal performance (optional but recommended)

## Installation

### Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

### Install the required Python Packages
```bash
pip install -r requirements.txt
pip install transformers
pip install bitsandbytes
pip install -qU langchain_community pypdf
pip install -qU "langchain-chroma>=0.1.2"
pip install -U sentence-transformers
pip install langchain==0.0.174 -i https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com
```

## Usage
Load a PDF Document: Place your PDF document in the ./data/ directory. The script is currently set to load falcon-users-guide-2021-09-compressed.pdf.

Run the Script: Execute the script to perform a similarity search and generate a response.

bash
Download
Copy code
python <script-name>.py
Enter a Prompt: When prompted, enter a query or problem statement. The script will search for relevant content in the loaded document and generate a response.

## How It Works
- Document Loading: The PyPDFLoader loads the specified PDF document and extracts its content.
- Embeddings: The HuggingFaceEmbeddings model generates embeddings for the document content.
- Vector Store: The Chroma vector store saves the document embeddings and facilitates similarity searches.
- Similarity Search: The script performs a similarity search using the user's prompt to find the most relevant document content.
- Response Generation: A language model pipeline generates a context-aware response based on the search results and user prompt.
#### Configuration
- Model Configuration: You can change the model used for embeddings and response generation by modifying the model_name and model_kwargs parameters.
- Document Path: Update the path in PyPDFLoader to load a different PDF document.
- Vector Store Directory: Change the persist_directory in Chroma to specify where to save the vector store data.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- Hugging Face Transformers
- Langchain
- Chroma

##  Contact
For any questions or feedback, please contact Mark at mark@mindifyai.dev