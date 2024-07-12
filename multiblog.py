import os
import logging
import requests
from bs4 import BeautifulSoup
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")

logger.info(f"Azure OpenAI Endpoint: {azure_openai_endpoint}")

# Set the environment variables in the script if not set already
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint
os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key

# Verify environment variables are loaded
if not azure_openai_endpoint or not azure_openai_key:
    raise ValueError("Environment variables for Azure OpenAI endpoint or key are not set.")

# Step 1: Fetch content from an online blog link
blog_url = "https://pinchofyum.com/quick-homemade-ramen"  
response = requests.get(blog_url)

if response.status_code != 200:
    raise ValueError(f"Failed to fetch content from {blog_url}")

soup = BeautifulSoup(response.content, 'html.parser')
blog_content = soup.get_text(separator="\n")

# Check the fetched content length
logger.info(f"Fetched blog content length: {len(blog_content)} characters")

# Step 2: Create a LangChain Document object for the blog content
document = Document(page_content=blog_content)

# Step 3: Create embeddings using Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-03-15-preview"
)

# Step 4: Create a vector store from the document
vector_store = FAISS.from_documents([document], embeddings)

# Step 5: Create a conversational retrieval chain with a custom prompt
prompt_template = """
You are an AI-driven bot. Your task is to answer queries based on the provided documents. Your responses should strictly adhere to the information presented in the documents, providing a seamless flow of information from the document to the user. Your responses must be Spartan—brief and direct. Do not engage with abusive queries.
Please provide concise yet comprehensive answers.
Don't justify your answers. Don't give information not mentioned in the CONTEXT INFORMATION. If the answer is not in the context, say the words "Sorry, I am unable to answer your question with the information available to me"

Query: {question}

Context:
{context}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

llm = AzureChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    azure_deployment='gpt-35-turbo',
    openai_api_version="2023-03-15-preview"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    verbose=True,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt
    }
)

while True:
    query = input("Please enter your query (type 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        print("Thank you for using the bot!")
        break
    
    try:
        response = qa_chain({
            "query": query
        })
        answer = response["result"]
        print(answer)
    except Exception as e:
        logger.error(f"Could not complete the querying process. Error: {e}")
        print("Some issue occurred, please try again")
