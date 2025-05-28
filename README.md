# Offline-RAG-Chatbot
This project is a Retrieval-Augmented Generation (RAG) chatbot built with Python and Streamlit, designed to answer questions about uploaded PDF or text files. It uses a local OllamaLLM (`phi` model) for text generation and a FAISS vector store with HuggingFaceEmbeddings (`all-MiniLM-L6-v2`) for document retrieval, which runs fully offline.


## Features
- **Offline Operation**: Runs entirely locally after downloading models, ideal for environments without internet.
- **Document Support**: Processes PDF and text files using `PyPDFLoader` and `TextLoader`.
- **RAG Pipeline**: Combines retrieval (FAISS vector store) with generation (Ollama `phi`) for accurate, context-based answers.
- **Streamlit Frontend**: Intuitive interface for uploading files, viewing document details, and querying content.
- **Error Handling**: Robust file validation and detailed error messages for debugging.

## Prerequisites
- **Python**: 3.8 or higher
- **Ollama**: Installed and running locally
- **Hardware**: Sufficient memory for `phi` (2-4GB RAM/VRAM) and embeddings

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -U langchain langchain-community langchain-ollama sentence-transformers transformers faiss-cpu streamlit pypdf
   ```

2. **Install Ollama**:
   - Follow instructions at [Ollama](https://ollama.com/) to install.
   - Pull the `phi` model:
     ```bash
     ollama pull phi
     ```

3. **Download Embedding Model** (requires internet, run once):
   ```python
   from sentence_transformers import SentenceTransformer
   model_name = "all-MiniLM-L6-v2"
   save_path = "./models/all-MiniLM-L6-v2"
   model = SentenceTransformer(model_name)
   model.save(save_path)
   ```

## Usage
1. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

2. **Run the Chatbot**:
   ```bash
   streamlit run chatbot3.py
   ```

3. **Interact with the Chatbot**:
   - Open the Streamlit app in your browser (default: `http://localhost:8501`).
   - Upload a `.txt` or `.pdf` file.
   - Enter a question about the document content.
   - View the answer and source excerpts in the interface.

## Project Structure
- `chatbot3.py`: Main script with RAG pipeline and Streamlit interface.
- `models/all-MiniLM-L6-v2/`: Directory for pre-downloaded embedding model.
- `README.md`: Project documentation.

## How It Works
1. **Document Processing**:
   - Uploads are saved temporarily with correct extensions (e.g., `temp_file.pdf`).
   - Documents are loaded and split into 500-character chunks with 100-character overlap.
2. **Embedding & Retrieval**:
   - Chunks are converted to embeddings using a local `all-MiniLM-L6-v2` model.
   - FAISS indexes embeddings for fast similarity search.
3. **Answer Generation**:
   - Top 2 relevant chunks are retrieved for each query.
   - The `phi` model generates answers using a custom prompt template.
4. **Frontend**:
   - Streamlit displays file details, processing status, answers, and source excerpts with expanders.

## Troubleshooting
- **Error: "MaxRetryError"**:
  - Ensure the embedding model is saved in `./models/all-MiniLM-L6-v2`.
  - Verify `local_files_only=True` in `create_vector_store`.
- **Error: "File not found"**:
  - Check write permissions in the project directory.
  - Use valid `.txt` (UTF-8) or non-encrypted `.pdf` files.
- **Ollama Issues**:
  - Confirm the server is running (`curl http://localhost:11434`).
  - Verify `phi3` is listed (`ollama list`).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.


## Acknowledgments
- Built with [LangChain](https://python.langchain.com/), [Ollama](https://ollama.com/), and [Streamlit](https://streamlit.io/).
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `phi` for generation.
