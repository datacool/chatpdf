__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
#from langchain import hub 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import PromptTemplate
import streamlit as st
import pandas as pd
import tempfile
import os
from io import StringIO

from dotenv import load_dotenv
from pathlib import Path

env_path = Path("../.env")
load_dotenv(dotenv_path=env_path)

st.title("ChatPDF")
st.write("---")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path,"wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    return pages 

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
        

#Loader
#loader = PyPDFLoader("unsu.pdf")
#pages = loader.load_and_split()

#Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

#Embedding
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    
)

#Chroma DB
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

db = Chroma.from_documents(texts, embeddings_model)

#User Input
st.header("PDF에게 질문해보세요!!")
question = st.text_input("질문을 입력하세요:")

if st.button("질문하기"):
    with st.spinner("답변 생성 중... 잠시만 기다려주세요."):
        #Retriever
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        retriever = db.as_retriever()
        #docs = retriever.invoke(question)

        #Prompt Template
        #prompt = hub.pull("rlm/rag-prompt")
        #Prompt Template
        # hub.pull 대신 RAG 표준 PromptTemplate을 직접 정의합니다.
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "Thanks for asking!" at the end of the answer.

        {context}

        Question: {question}
        Helpful Answer:"""

        prompt = PromptTemplate.from_template(template) # <--- 'prompt' 변수 정의

        #Generate
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()    
        )
        
        #Question
        #question = "아내가 먹고 싶어하는 음식은 무엇이야?"
        result = rag_chain.invoke(question)
        st.write(result)
        print(result)

        




