# streamlit run main.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llm_handler.ollama_handler import OllamaModelHandler
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "prompt.txt")

with open(file=PROMPT_TEMPLATE_PATH, mode="r", encoding="utf8") as file:
    prompt_template = file.read()

ollama_handler = OllamaModelHandler()
llm = ollama_handler.get_model_instance()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_template,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("AI Test Case")

for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Your Question")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    response = chain_with_history.stream(
        {"input": question}, config={"configurable": {"session_id": "any"}}
    )
    with st.chat_message("assistant"):
        st.write_stream(response)
