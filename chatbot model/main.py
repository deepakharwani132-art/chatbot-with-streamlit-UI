import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Agent Chatbot (Groq)", layout="wide")
st.title("🤖 AI Agent Chatbot By Deepak Kumar Harwani")

groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# -----------------------------
# GROQ MODEL
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# -----------------------------
# TOOLS FOR THE AGENT
# -----------------------------
def weather_tool(city: str):
    return f"Fake weather report: It's always sunny in {city} 🌞."

tools = [
    Tool(
        name="WeatherTool",
        func=weather_tool,
        description="Returns weather information for a city"
    )
]

# -----------------------------
# MEMORY
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -----------------------------
# AGENT INITIALIZATION
# -----------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# -----------------------------
# CHAT UI (FIXED)
# -----------------------------
st.subheader("Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Message input (only ONE box — does not duplicate)
user_input = st.chat_input("Ask something...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.run(user_input)
                st.write(response)
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)

    # Save agent response
    st.session_state.messages.append({"role": "assistant", "content": response})
