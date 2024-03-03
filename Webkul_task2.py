import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import ollama
import gradio as gr
import textwrap
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

from langchain_community.embeddings import OllamaEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import textwrap
from langchain.llms import Ollama
from dotenv import load_dotenv
import os


embeddings = embeddings = OllamaEmbeddings(model='phi')
new_db = Chroma(persist_directory = "./chroma_db",embedding_function=embeddings)

print("About to give output")
# query = "What is machine learning"
# docs = new_db.similarity_search_with_score(query)
# print(docs)

prompt = """
### System:
You are an AI Assistant that follows instructions extreamly well. \
Help as much as you can.

### User:
{prompt}

### Response:

"""

template = """
### System:
You are an respectful and honest assistant.

### Context:
{context}

### User:
{question}

### Response:
"""

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )


retriever = new_db.as_retriever()
llm = Ollama(model="phi", temperature=0)

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)


# Prettifying the response
def get_response(query):
    # Creating the chain
    chain = load_qa_chain(retriever, llm, prompt)

    # Getting response from chain
    response = chain({'query': query})

    return response['result']

# print(get_response("What is semi-supervised learning ?"))
iface = gr.Interface(
    fn=get_response,
    inputs="text",
    outputs="text",
    title="Rag Chain Based Chatbot",
    description="This is a chatbot based on RAG technique"
)
iface.launch()