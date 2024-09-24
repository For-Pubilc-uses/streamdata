import os
import pinecone
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import PineconeHybridSearchRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from prompts import Prompt


class RAGImplementation:
    def __init__(self):
        load_dotenv()
        
        os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv('openai_api_key'))
        self.vectorstore = PineconeVectorStore.from_existing_index(index_name="steamdata", embedding=self.embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('openai_api_key'))
        
        self.setup_sparse_encoder()
        self.setup_retriever()
        self.setup_rag_chain()

    def setup_sparse_encoder(self):
        # Load the pre-trained BM25Encoder
        self.sparse_encoder = BM25Encoder().load('bm25_encoder.json')

        # Add the sparse_encoder to the vectorstore
        self.vectorstore.sparse_encoder = self.sparse_encoder

    def setup_retriever(self):
        # Set up PineconeHybridSearchRetriever
        pc = pinecone.Pinecone(api_key=os.getenv('pinecone_api_key'))
        pinecone_index = pc.Index("steamdata")
        
        # Adjust the top_k and alpha parameters to balance between sparse and dense retrieval
        self.retriever = PineconeHybridSearchRetriever(
            index=pinecone_index,
            sparse_encoder=self.sparse_encoder,
            embeddings=self.embeddings,
            top_k=4,
            alpha=0.3
        )

    def format_docs(self, docs):
        return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

    def setup_rag_chain(self):
        prompt = ChatPromptTemplate.from_template(Prompt.template)
        
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    def ask(self, question: str):
        response = self.rag_chain_with_source.invoke(question)
        return response['answer']

if __name__ == "__main__":
    rag_implementation = RAGImplementation()
    print(rag_implementation.ask("tell me some info for dota2"))