import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import OllamaEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter


def file_processor(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  
    embedidngs = OllamaEmbeddings(model="deepseek-r1:latest")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma(persist_directory="./test_db", embedding_function=embedidngs)
    
    vectorstore.add_documents(chunks)
    print("File processed")

if __name__ == "__main__":
    pdf_path = "data/GabrielCurriculo.pdf"
    if(os.path.exists(pdf_path)):
        file_processor(pdf_path)
    else:
        print("File not found")    