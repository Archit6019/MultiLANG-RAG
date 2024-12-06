# MultiLangRAG System

MultiLangRAG is a multilingual document processing and question-answering system that combines vector search capabilities with large language models to provide accurate, context-aware responses.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/embaas/MultiLangRAG.git
cd MultiLangRAG
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

Add Tesseract to your system PATH:

```bash
export PATH="/usr/local/opt/tesseract/bin:$PATH"
```

5. Install Docker:

```bash
brew install docker
```

Make sure docker daemon is running, you can check this by running `docker ps`. Also add docker to your PATH:


## Configuration

1. Set up environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export QDRANT_URL="http://localhost:6333" - This is where qdrant will run locally, make sure docker daemon is running for this to work
```

## Running the System

1. Start the application:

```bash
python app/Gradio.py
```

2. Open your web browser and navigate to http://localhost:7860


## Using the System

### 1. Create a Collection
- Go to the "Create Collection" tab
- Enter a collection name
- Set vector size to 768 (this is the default size for the multilingual-e5-base embedder)
- Click "Create Collection"

### 2. Upload Documents
- Navigate to "Upload Document" tab
- Select your collection (The same collection you created in step 1)
- Upload a PDF file
- Provide a document name
- Click "Upload Document"

### 3. Chat with Documents
- Go to the "Chat" tab
- Enter your collection name (The same collection you created in step 1)
- Type your question
- View AI responses and source documents






