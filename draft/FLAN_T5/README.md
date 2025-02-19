This application setup will enable the user to query the GNA project content and get generated responses from the RAG system.

Run the application directly: `python app.py`
This script integrates everything into a command-line or API interface.

Breakdown instructions to run the system separately:
1. fetch and parse data (run `python fetch_data.py`)
2. Preprocess content (run `python preprocess_data.py`)
3. Create index (run `python index_data.py`)
4. Query the content (run `python query_system.py`)