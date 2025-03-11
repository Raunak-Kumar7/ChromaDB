from fastapi import FastAPI
import logging
from app.api import db_api
from app.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ChromaDB API Server",
    description="API server for managing embeddings and database operations",
)

app.include_router(db_api.router, tags=["vectordb"])