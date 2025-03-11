import uvicorn

if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        workers=1,  # Use a single worker as locks are used in ChromaDB
        # reload=True
    )