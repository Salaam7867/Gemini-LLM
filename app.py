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

# Initialize Gemini LLM (cached to avoid recreation on every rerun)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

llm = get_llm()

# Persistent memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Store chat messages for display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful AI assistant.
Answer clearly and concisely.

Conversation history:
{history}

User: {input}
Assistant:"""
)

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input using chat_input (better than text_input for chat apps)
if user_input := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=user_input)
            st.write(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a clear chat button in the sidebar
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This is a chat interface powered by Google's Gemini AI model using LangChain.")
    st.markdown("Ask me anything!")