import uvicorn
from app.api.db_api import app  # Import the FastAPI app from api.py
import logging

# Configure logging
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO) 

if __name__ == "__main__":
    address = "127.0.0.1"
    port = 8002

    logger.info(f"Starting FastAPI server at http://{address}:{port}/api")
    # Use Single worker. This is needed as locks are used to perform operations on ChromDB
    uvicorn.run("app.api.db_api:app", host=address, port=port, workers=1,
                # reload=True
                )