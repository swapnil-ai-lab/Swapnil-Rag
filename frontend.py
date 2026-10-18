import streamlit as st
import backend as demo

st.set_page_config(page_title="OTC Lifecycle Assistant", page_icon="📌", layout="centered")
st.title("📌 OTC Lifecycle Assistant")
st.caption("RAG + LangChain + Pinecone, grounded in the OTC website")

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_retriever():
    return demo.build_retriever()

try:
    retriever = get_retriever()
except Exception as exc:
    st.error(f"Could not connect to Pinecone or prepare the website index: {exc}")
    st.stop()

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory = demo.demo_memory()
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Re-index website", use_container_width=True):
        with st.spinner("Re-scraping OTC websites and upserting into Pinecone..."):
            n_chunks = demo.index_website(reset=True)
        st.success(f"Indexed {n_chunks} chunks into Pinecone.")

    st.divider()
    st.markdown(
        "**About**\n\n"
        "- Retrieval-augmented over OTC websites\n"
        f"- Vector DB: Pinecone (`{demo.PINECONE_INDEX_NAME}`)\n"
        "- Few-shot + chat memory\n"
        "- Model: `gpt-4o-mini`"
    )

if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Ask about Trade confirmations, settlements, or lifecycle management...")
if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, updated_memory = demo.demo_conversation(
                input_text=input_text,
                memory=st.session_state.memory,
                retriever=retriever,
            )
        st.markdown(response)

    st.session_state.memory = updated_memory
    st.session_state.chat_history.append({"role": "assistant", "text": response})
