from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings
import uvicorn

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Setup AWS and Bedrock client
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name='us-east-1')

def create_embeddings(client):
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=client)

def create_bedrock_llm(client):
    return ChatBedrock(model_id='anthropic.claude-3-sonnet-20240229-v1:0', client=client, model_kwargs={'temperature': 0}, region_name='us-east-1')

# Initialize everything
bedrock_client = get_bedrock_client()
bedrock_embeddings = create_embeddings(bedrock_client)
vectorstore = PineconeVectorStore(index_name='production-ready-tutorial', embedding=bedrock_embeddings, pinecone_api_key={pinecone_api_key})
model = create_bedrock_llm(bedrock_client)

template = '''
Use the following context to answer the question:
{context}

Question: {question}'''

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": vectorstore.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
add_routes(
    app,
    chain,
    path="/knowledge",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)