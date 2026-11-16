
import streamlit as st

from agent import MovieAgent
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(
    page_title="Movie GraphRAG",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie GraphRAG")
st.caption("Explore Bollywood movies using AI, Pinecone and Neo4j.")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = MovieAgent()

if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.header("Movie Assistant")

    st.write("Powered by Gemini, Pinecone and Neo4j.")

    st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    st.subheader("Try asking")

    examples = [
        "Movies directed by Nila Madhab Panda",
        "Recommend movies about poverty",
        "Who directed I Am Kalam?",
        "Find movies about social inequality"
    ]

    for example in examples:
        st.caption("• " + example)


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input("Ask something about Bollywood movies...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        try:
            with st.spinner("Searching movie database..."):

                answer = st.session_state.agent.ask(
                    user_input,
                    st.session_state.history
                )

            if isinstance(answer, list):
                answer = "\n".join(
                    block.get("text", "")
                    for block in answer
                    if isinstance(block, dict)
                )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:

            st.error(f"Something went wrong: {error}")
