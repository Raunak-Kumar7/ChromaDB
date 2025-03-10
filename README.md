# ChromaDB API Server

## Overview
The ChromaDB API Server is a FastAPI-based vector database service designed for managing embeddings efficiently. It supports adding, deleting, and querying vectorized data while integrating with LLMs for advanced retrieval-augmented generation (RAG) workflows.

## Dev guide

Use Python 3.9 or later To run the app:

    $ cd ChromaDB
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install --pre -r requirements.txt
    $ python run.py

You can confirm the service is up by doing this:

    $ curl 'http://localhost:8002/'

You should see a result like this:  `Welcome To ChromaDB!`

## API Endpoint

    $ /api/v1/alloperations