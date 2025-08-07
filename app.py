import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv



#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A chatbot with OpenAI"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "{question}")
])

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

## title
st.title("Enhanced qa chatbot with openAI")

## sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("enter open ai api key:", type="password")

# Drop down
llm = st.sidebar.selectbox("select open ai model",["gpt4o","gpt4-turbo","gpt-4"])

# adjust response parameter
temperature = st.sidebar.slider("Temp", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max token", min_value=50, max_value=300, value=150)

# Main interface
st.write("ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("provide query")