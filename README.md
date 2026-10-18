# OTC Lifecycle Assistant

A small Streamlit chat app that answers questions about OTC trade confirmations, settlement, reconciliation, and lifecycle management. It uses retrieval-augmented generation (RAG) to ground responses in selected DTCC and SWIFT web pages.

## What it does

- Retrieves relevant passages from the indexed reference pages and uses OpenAI `gpt-4o-mini` to answer questions.
- Stores page embeddings in Pinecone using OpenAI `text-embedding-3-small` embeddings.
- Keeps chat history for the current Streamlit session.
- Lets you clear the conversation or re-index the reference pages from the sidebar.

On first launch, the app creates the Pinecone index if needed and indexes the pages when its namespace is empty. Re-indexing replaces the contents of the app's Pinecone namespace.

## Requirements

- Python 3.14 (the version specified by this project)
- An OpenAI API key
- A Pinecone API key

## Run locally

1. Clone the repository and open a terminal in the project folder.
2. Create and activate a virtual environment, then install the dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r req.txt
   ```

   On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

3. Create a `.env` file in the project folder and add your keys:

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

   The default Pinecone index is `swapnil-rag`, using AWS in `us-east-1`. You can override these defaults in `.env` with `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and `PINECONE_REGION`.

4. Start the app:

   ```bash
   streamlit run frontend.py
   ```

## Reference pages

- [DTCC Trade Information Warehouse](https://www.dtcc.com/products-and-services/repository-derivatives-services/trade-information-warehouse)
- [SWIFT Settlement and Reconciliation](https://www.swift.com/securities/settlement-and-reconciliation)
- [SWIFT Standardised Business Flows](https://www.swift.com/securities/securities-market-infrastructure/standardised-business-flows)
- [SWIFT Securities View](https://www.swift.com/products/swift-securities-view)

## GitHub and hosting

The `.env` file is excluded by `.gitignore`; keep API keys out of commits. GitHub can host the source repository, but running the Streamlit app online requires a hosting service configured with the same environment variables.
