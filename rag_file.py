from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings

class RAGFile: 
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="deepseek-r1:latest")

        self.vectorstore = Chroma(persist_directory="./test_db", embedding_function=self.embeddings)

        self.llm = Ollama(model="deepseek-r1:latest")  

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever()
        )

    def ask(self, query):
        return self.qa_chain.run(query)

if __name__ == "__main__":
    rag = RAGFile()
    answer = rag.ask("Aonde Gabriel Antonio Pereira de Camargos tem experiencias de trabalho?")
    print("Resposta:", answer)
