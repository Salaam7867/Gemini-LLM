import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="GenAI Gemini Chat", page_icon="🤖")
st.title("GenAI Chat Assistant (Gemini)")

# Initialize Gemini LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("OPENAI_API_KEY")
)


# Persistent memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful AI assistant.
Answer clearly and concisely.

Conversation history:
{history}

User:
{input}

Assistant:
"""
)

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt
)

# User input
user_input = st.text_input("You:")

# Generate response
if user_input:
    response = conversation.predict(input=user_input)
    st.write("AI:", response)
