import boto3
import streamlit as st
from langchain_community.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


#Bedrock client setup
bedrock= boto3.client(service_name='bedrock-runtime', 
                      region_name=region_name,
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)

#Get embedding model from Bedrock
bedrock_embeddings= BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock) #titan-embed-text-v1 model for embedding

def get_documents():
    loader= PyPDFDirectoryLoader("data")
    documents= loader.load()
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                  chunk_overlap=500)
    docs=text_splitter.split_documents(documents)
    return docs


def get_vectorstore(docs):
    vectorstore_faiss=FAISS.from_documents(embedding=bedrock_embeddings, documents=docs)
    vectorstore_faiss.save_local("faiss_local") #save locally in the current directory


def get_llm():
    llm = Bedrock(model_id = "mistral.mistral-7b-instruct-v0:2", client = bedrock) #connect mistral-7b-instruct-v0 model for LLM through Bedrock API
    return llm

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_llm_response(llm, vectorstore_faiss, query):
    #to connect llm to the knowledge base for QA operation
    qa= RetrievalQA.from_chain_type(llm= llm, 
                                    chain_type= "stuff", #since QA, we use stuff chain
                                    retriever= vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}), #connect to vectorstore for retrieval
                                    return_source_documents= True, #return the source documents for detailed answer
                                    chain_type_kwargs={"prompt": PROMPT}) #set the prompt template

    response= qa({"query": query}) #pass the question to the LLM
    return response['result'] #return the answer


#to run streamlit app run "streamlit run main.py"
def main():
    st.set_page_config(page_title="PDF QA")
    st.header("End to End RAG using Bedrock, LangChain, and Streamlit")
    
    user_question = st.text_input("Ask a question from the llama2 paper")

    with st.sidebar:
        st.title("Update and create the vectorstore") #will create a sidebar

        if st.button("Store vectorstore"):  #create a button for triggering the vectorstore creation
            with st.spinner("Creating vectorstore..."):
                docs=get_documents()
                get_vectorstore(docs)
                st.success("Vectorstore created successfully!")

        if st.button("Send"):  #
            with st.spinner("Loading FAISS vectorstore..."):
                faiss_index= FAISS.load_local("faiss_local",bedrock_embeddings, allow_dangerous_deserialization=True) #load the vectorstore from local directory
                llm= get_llm()
                st.write(get_llm_response(llm,faiss_index, user_question)) #write the response to the user
                

if __name__== "__main__":
    main()


