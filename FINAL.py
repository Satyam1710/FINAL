import ollama
import gradio as gr
import PyPDF2
import pdf2image
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFacePipeline

import transformers
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"
# print("This process is done")

# loader = PyPDFLoader(r"D:\testing.pdf")
loader = PyPDFLoader(r"D:\FINAL.txt.pdf")
pages = loader.load_and_split()
# print(pages[11])

# Embedding
embeddings = HuggingFaceEmbeddings()

#Vectordb
vectordb = Chroma.from_documents(pages , embeddings)

# Similarity search
query = "What is maachine learning"
docs = vectordb.similarity_search(query)
# print("THIS IS THE ANSWER = ",docs)

#For RAG

# create the retriever
# retriever = vectordb.as_retriever()

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


model_list = ollama.list()
model_names = [model['model'] for model in model_list['models']]

def chat_ollama(user_input, history, Model):
    # retrieved_docs = retriever.invoke(user_input)
    # user_input = format_docs(retrieved_docs)

    stream = ollama.chat(
        model=Model,
        messages=[
                {
                    'role': 'user', 
                    'content': user_input
                },
            ],
        stream=True,
    )
    

    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def rag_chain(question,history,model,):
#     retrieved_docs = retriever.invoke(question)
#     formatted_text = format_docs(retrieved_docs)
#     response = chat_ollama()
#     return formatted_text

with gr.Blocks(title="Ollama Chatbot", fill_height=True) as demo:
    gr.Markdown("# Ollama Chatbot")
    # model_list = gr.Dropdown(model_names, value="llama2:latest", label="Model", info="Model to chat with")
    model_list = gr.Dropdown(model_names, value="phi", label="Model", info="Model to chat with",allow_custom_value=True)
    gr.ChatInterface(chat_ollama, additional_inputs=model_list)

if __name__ == "__main__":
    demo.launch()