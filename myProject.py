import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
# import textwrap
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

loader = PyPDFLoader(r"D:\FINAL.txt.pdf")
docs = loader.load()

# # print(docs)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(chunks,embeddings)

query="What is support-vector machine"
ans=vectorstore.similarity_search(query)
# print(ans)

retriever = vectorstore.as_retriever()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)    

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    
    docs = retrieved_docs
    
    prompt_template ="""
    You are a Q&A assistant. Your goal is to answer all questions.\n\n

    Context:\n {context}?\n
    Question: \n{question}\n
     """


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain.run(input_documents=docs,question=question,return_only_outputs = True)
    # Comment below this if getting an error
    if response[:29] =="The provided context does not" or response[:29] =="This context does not mention" :
        stream = ollama.chat(
           model='phi',
           messages=[{'role': 'user', 'content': question }],
           stream=True,
           )
        # return stream
        ans =""
        for chunk in stream:
           ans += chunk['message']['content']
           yield ans
    else:
        yield response
        # ans =""
        # for chunk in response:
        #    ans += chunk['message']['content']
        #    yield ans

# ans = user_input("Who to implement machine learning model")
# print(ans)
        
# gen_object = user_input("who is the prime minister of india")

# Iterate over the generator
# for value in gen_object:
#     print(value)



iface = gr.Interface(
    fn=user_input,
    inputs="text",
    outputs="text",
    title="Rag Chain Based Chatbot",
    description="This is a chatbot based on RAG technique"
)
iface.launch()