import os
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st

# 1. Load the Phi model using Ollama
def load_phi_model():
    llm = OllamaLLM(
        model="phi",
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=512,
    )
    return llm

# 2. Load and process the uploaded document
def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file extension for {file_path}. Use .pdf or .txt.")
        documents = loader.load()
        return documents
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")

# 3. Chunk the document
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# 4. Create embeddings and vector store
def create_vector_store(chunks):
    # Load embeddings from local directory
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/all-MiniLM-L6-v2",  # Path to local model
        model_kwargs={"local_files_only": True}  # Ensure offline mode
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ... (rest of the functions: prompt_template, create_rag_chain, answer_query, command_line_interface remain unchanged)

# 5. Define prompt template for RAG
def prompt_template():
    template = """Use the following context to answer the question. Only use information from the context, and do not make up answers:
    {context}
    Question: {question}
    Answer: """
    return PromptTemplate(template=template, input_variables=["context", "question"])

# 6. Create RAG pipeline
def create_rag_chain(vector_store, llm):
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template()}
    )
    return rag_chain

# 7. Handle user queries
def answer_query(rag_chain, query):
    result = rag_chain({"query": query})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources

# 8. Command-line interface
def command_line_interface():
    llm = load_phi_model()
    file_path = input("Enter the path to your text or PDF file: ")
    
    try:
        documents = load_document(file_path)
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
        rag_chain = create_rag_chain(vector_store, llm)
        
        print("\nChatbot is ready! Ask questions about the document.")
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break
            try:
                answer, sources = answer_query(rag_chain, query)
                print("\nAnswer:", answer)
                print("\nSources:")
                for i, doc in enumerate(sources, 1):
                    print(f"Source {i}: {doc.page_content[:200]}...")
            except Exception as e:
                print(f"Error processing query: {e}")
    except Exception as e:
        print(f"Error processing file: {e}")

# 9. Streamlit web interface
def streamlit_interface():
    st.title("Local Phi Chatbot (Powered by Ollama)")
    st.write("Upload a text or PDF file and ask questions based on its content.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    
    if uploaded_file:
        # Determine file extension based on uploaded file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_path = f"temp_file{file_extension}"
        
        try:
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:
                content = uploaded_file.read()
                st.write(f"Saving file: {uploaded_file.name}, Size: {len(content)} bytes")
                f.write(content)
            
            # Verify file was saved
            if not os.path.exists(file_path):
                st.error(f"Failed to save temporary file at {file_path}")
                return
            
            try:
                # Load model and process document
                llm = load_phi_model()
                documents = load_document(file_path)
                chunks = chunk_documents(documents)
                vector_store = create_vector_store(chunks)
                rag_chain = create_rag_chain(vector_store, llm)
                
                # Query input
                query = st.text_input("Ask a question about the document:")
                if query:
                    try:
                        with st.spinner("Processing query..."):
                            answer, sources = answer_query(rag_chain, query)
                        st.write("**Answer:**", answer)
                        st.write("**Sources:**")
                        for i, doc in enumerate(sources, 1):
                            st.write(f"Source {i}: {doc.page_content[:200]}...")
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        st.warning(f"Failed to delete temporary file: {e}")
        except Exception as e:
            st.error(f"Error saving file: {e}")

# 10. Main execution
if __name__ == "__main__":
    streamlit_interface()