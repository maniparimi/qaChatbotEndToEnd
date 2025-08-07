import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import SomeVectorStore
from langchain.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

import os
groq_api_key = os.getenv('GROQ_API_KEY')

model_name = 'llama3-8b-8192'
# or model_name = 'gamma-7b-it'

chat_prompt_template = ChatPromptTemplate.from_template(
    """
    Hey, answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    Context: {context}
    Question: {question}
    """
)

def create_vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state['embeddings'] = embeddings
        st.session_state['loader'] = PyPDFDirectoryLoader('research_paper')
        st.session_state['documents'] = st.session_state['loader'].load()
        st.session_state['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state['final_documents'] = st.session_state['text_splitter'].split_documents(
            st.session_state['documents'][:50]
        )
        st.session_state['vectors'] = FAISS.from_documents(
            st.session_state['final_documents'],
            st.session_state['embeddings']
        )


st.session_state['vectors'] = FAISS.from_documents(
    st.session_state['final_documents'],
    st.session_state['embeddings']
)


user_prompt = st.text_input('Enter your query from the documents or from the research paper')
if st.button('Document Embedding'):
    create_vector_embedding()
    st.write('Your vector database is ready')


document_chain = create_stuff_document_chain(llm, chat_prompt_template)
retriever = st.session_state['vectors'].as_retriever()
retrieval_chain = create_retriever_chain(retriever, document_chain)


import time
start_time = time.process_time()
response = retrieval_chain.invoke({'input': user_prompt})
response_time = time.process_time() - start_time
st.write(f'Response time is {response_time}')


st.write(response['answer'])
with st.expander('Document similarity search'):
    for i, doc in enumerate(response['context']):
        st.write(doc.page_content)



from openai.embeddings_utils import get_embedding
openai_api_key = os.getenv('OPENAI_API_KEY')


