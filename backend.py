import os
import time
import hashlib
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "rag-pinecone/0.1.0")

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

OTC_URLS = [
     # OTC derivatives lifecycle, confirmation and settlement
    "https://www.dtcc.com/products-and-services/repository-derivatives-services/trade-information-warehouse",

    # Securities settlement and reconciliation
    "https://www.swift.com/securities/settlement-and-reconciliation",

    # Clearing, settlement, reconciliation and post-trade matching
    "https://www.swift.com/securities/securities-market-infrastructure/standardised-business-flows",

    # Securities settlement tracking and exceptions
    "https://www.swift.com/products/swift-securities-view",
]


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "swapnil-rag")
PINECONE_NAMESPACE = "swapnil-website"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # must match the embedding model above


def demo_chatbot():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Create a .env file with: "
            "OPENAI_API_KEY=your_key_here"
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_pinecone_index():
    """Connect to the Pinecone index, creating it on first run."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found. Add it to your .env file: "
            "PINECONE_API_KEY=your_key_here"
        )

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


def namespace_vector_count(index):
    """Number of vectors already stored in our namespace (0 if none)."""
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    ns = namespaces.get(PINECONE_NAMESPACE)
    return ns.vector_count if ns else 0


def make_chunk_ids(chunks):
    """
    Deterministic IDs (source URL + chunk position), so re-indexing
    overwrites existing records instead of creating duplicates.
    """
    ids, counters = [], {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        i = counters.get(source, 0)
        counters[source] = i + 1
        ids.append(hashlib.md5(f"{source}::{i}".encode()).hexdigest())
    return ids


def index_website(urls=OTC_URLS, reset=True):
    """
    Scrape the URLs, chunk them, and upsert into Pinecone.
    reset=True clears the namespace first so removed/shortened
    pages don't leave stale chunks behind.
    Returns the number of chunks upserted.
    """
    docs = WebBaseLoader(urls).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "No chunks were extracted from the source pages; "
            "the existing Pinecone namespace was not changed."
        )

    index = get_pinecone_index()

    if reset:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )
    vector_store.add_documents(chunks, ids=make_chunk_ids(chunks), batch_size=100)

    return len(chunks)


def build_retriever(urls=OTC_URLS, k=4):
    """
    Connect to Pinecone. Only scrapes + embeds if the namespace is empty,
    so app restarts no longer re-index the whole website.
    """
    index = get_pinecone_index()

    if namespace_vector_count(index) == 0:
        index_website(urls, reset=False)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )



examples = [
    {
        "input": "Who are you?",
        "output": "I'm the Swapnil's assistant. I help you with questions about our "
                  "Confirmation and Settlement processes and Lifecycle Management.",
    },
    {
        "input": "What are the common reasons for an OTC interest rate swap confirmation mismatch, and how should an operations analyst investigate the break?",
        "output": "Common OTC interest rate swap confirmation mismatches include differences in fixed rates, notional amounts, effective/maturity dates, payment frequencies, floating-rate indices, or day-count conventions. "
                  "To investigate, an operations analyst should first review the trade details in the confirmation against the original trade capture. They should check for discrepancies in the terms, verify the counterparties' records, and consult any relevant market data or documentation. Communication with the counterparty may be necessary to resolve any ambiguities or errors.",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an OTC Operations Copilot specializing in trade confirmations, settlement, reconciliation, and OTC derivatives. "
            "Answer questions concisely and professionally, preferably in 2–4 lines. Use the context retrieved from Pinecone to ground your answers. If the information is unavailable in the retrieved context, state that clearly rather than inventing facts. "
            "Provide practical operational guidance, identify discrepancies and potential risks, and recommend appropriate investigation or escalation steps. Never assume disputed trade details are correct without verification. Include source URLs when available. "
            "don't make things up. Match the tone of the examples & For More Queries, Reach OTC Team : Swapnil Madgulwar\n\n"
            "OTC website context:\n{context}",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def demo_memory():
    return InMemoryChatMessageHistory()


def demo_conversation(input_text, memory, retriever):
    try:
        llm = demo_chatbot()

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["input"]))
            )
            | CHAT_PROMPT
            | llm
        )

        chat_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: memory,
            input_messages_key="input",
            history_messages_key="history",
        )
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "Swapnil-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory
