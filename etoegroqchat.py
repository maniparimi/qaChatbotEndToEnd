import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory


import os
from dotenv import load_dotenv
load_dotenv()
#give token here
embeddings = HuggingFaceEmbeddings()

st.title('Conversational RAG with PDF uploads and chat history')
st.write('Upload PDFs and chat with the content.')

api_key = st.text_input('Enter your Groq API key', type='password')

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name='gemma2-9b-it')

session_id = st.text_input('Session ID', value='default_session')
if 'store' not in st.session_state:
    st.session_state['store'] = {}

uploaded_files = st.file_uploader('Choose a PDF file', type='pdf', accept_multiple_files=True)
documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_pdf = f'temp_{uploaded_file.name}'
        with open(temp_pdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        'Given a chat history and the latest user question which might reference context in the chat history, '
        'formulate a standalone application question which can be understood without the chat history. '
        'Do not answer the question, just reformulate it if needed and otherwise return this.'
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    answer_q_system_prompt = (
        'You are an assistant for the question answer task. Use the following pieces of retrieved context to answer the question. '
        'If you do not know the answer, say that you do not know. Use three sentences maximum and keep the answer concise.\n\n{context}'
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', answer_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session_id):
        if session_id not in st.session_state['store']:
            st.session_state['store'][session_id] = ChatMessageHistory()
        return st.session_state['store'][session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )


    user_input = st.text_input('Ask a question:')
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke({'input': user_input}, config={"configurable":{'session_id': session_id}})
        st.write('Session History:', session_history)
        st.write('Answer:', response['answer'])
    else:
        st.warning('Please enter the Groq API key.')

