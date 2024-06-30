import os
from langchain_community.document_loaders import S3FileLoader,S3DirectoryLoader,AmazonTextractPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore

os.environ['PINECONE_API_KEY'] = 'PINECONE_API_KEY'

def chunck_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def create_embeddings(region):
    bedrock_client = get_bedrock_client(region)
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                        client=bedrock_client)
    return bedrock_embeddings
    
def stuff_vectordatabase(chunks, bedrock_embeddings, index_name):
    docsearch = PineconeVectorStore.from_documents(chunks, bedrock_embeddings, index_name=index_name)
    return docsearch

def main():
    bucket_name = 'production-ready-tutorial'
    prefix = 'data/'

    loader = S3DirectoryLoader(bucket_name, prefix=prefix)
    data = loader.load()

    chunks = chunck_data(data)
    embeddings = create_embeddings(region='us-east-1')
    
    index_name = 'production-ready-tutorial'
    stuff_vectordatabase(chunks, embeddings, index_name)
    
if __name__ == "__main__":
    main()