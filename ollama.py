from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import StringOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

prompt = ChatPromptTemplate.from_messages([
    ("system", "Hey, you are a helpful assistant. Please respond to the user queries."),
    ("user", "{question}")
])

model = Ollama(model="mistral")
output_parser = StringOutputParser()

# Example function to generate a response
def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer
if __name__ == "__main__":
    st.title("Q&A Chatbot with Ollama and Open Source Models")
    user_input = st.text_input("Enter your question:")
    if user_input:
        response = generate_response(user_input)
        st.write(response)



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