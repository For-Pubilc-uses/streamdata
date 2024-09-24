import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

from dataCleaner import convert_df_to_documents
load_dotenv()

class PineconeUploader:
    """
    A class for uploading documents to Pinecone vector database using hybrid search.

    This class handles the process of loading data from a CSV file,
    converting it to document format, and uploading it to Pinecone using hybrid search.

    Attributes:
        pinecone_api_key (str): The API key for Pinecone.
        openai_api_key (str): The API key for OpenAI.
        embeddings (OpenAIEmbeddings): An instance of OpenAIEmbeddings for creating embeddings.
        sparse_encoder (BM25Encoder): An instance of BM25Encoder for sparse encoding.

    Methods:
        load_and_convert_data(csv_path): Loads data from a CSV file and converts it to documents.
        upload_to_pinecone(documents, index_name): Uploads documents to Pinecone vector database using hybrid search.
    """

    def __init__(self):
        self.pinecone_api_key = os.getenv('pinecone_api_key')
        self.openai_api_key = os.getenv('openai_api_key')
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                                           openai_api_key=self.openai_api_key)
        # bm25 encoder name
        self.sparse_encoder = BM25Encoder().load('bm25_encoder.json')

    def load_and_convert_data(self, csv_path):
        """
        Loads data from a CSV file and converts it to document format.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            list: A list of Document objects.
        """
        df = pd.read_csv(csv_path, encoding='latin1')
        return convert_df_to_documents(df)

    def upload_to_pinecone(self, documents, index_name):
        """
        Uploads documents to Pinecone vector database using hybrid search.

        Args:
            documents (list): A list of Document objects to be uploaded.
            index_name (str): The name of the Pinecone index to upload to.

        Returns:
            None
        """
        pc = Pinecone(api_key=self.pinecone_api_key)
        pinecone_index = pc.Index(index_name)

        retriever = PineconeHybridSearchRetriever(
            index=pinecone_index,
            sparse_encoder=self.sparse_encoder,
            embeddings=self.embeddings,
            top_k=3
        )

        texts = [doc.page_content for doc in documents]
        retriever.add_texts(texts)
        print("Documents uploaded to Pinecone using hybrid search")


if __name__ == "__main__":
    uploader = PineconeUploader()
    # only keep the neceessary columns , prepare the columns first 
    docs = uploader.load_and_convert_data('modified_cleaned_dataset.csv')
    uploader.upload_to_pinecone(docs, "steamdata")
