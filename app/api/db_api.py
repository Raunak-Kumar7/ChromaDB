from fastapi import FastAPI, HTTPException
import logging
from app.chromadb.chromaClient import ChromaCollector
from app.utils.data_processor import process_and_add_to_collector
import app.utils.parameters as parameters
from app.models.request_models import AddRequest, DeleteRequest, GetRequest, AllOpsRequest
import time
logger = logging.getLogger(__name__)

app = FastAPI()
# singleton instance in your ChromaCollector class is to ensure that All requests to your ChromaDB server share the same instance of ChromaCollector. 
# A singleton ensures only one connection is maintained.
# The collector will be initialized as soon as your FastAPI app starts
collector = ChromaCollector()

# Disable StandAlone API Calls, Locks Removed from individual API functions
# ---------------------------------------------------------------------------------------------------------------
# @app.post("/api/v1/add")
# def add_data(request: AddRequest):
#     try:
#         process_and_add_to_collector(request.corpus, collector, request.clear_before_adding, request.metadata)
#         return {"message": "Data successfully added"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/v1/get")
# def get_data(request: GetRequest):
#     try:
#         n_results = request.n_results or parameters.get_chunk_count() # "default": 250,
#         max_token_count = request.max_token_count # or parameters.get_max_token_count()
#         results = collector.get_sorted_by_dist(request.search_strings, n_results, max_token_count, metadata=request.metadata)
        
#         return {"results": results}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/v1/delete")
# def delete_data(request: DeleteRequest):
#     try:
#         collector.delete(ids_to_delete=None, where=request.metadata) # Deletion is based on Metadata only
#         return {"message": "Data successfully deleted"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/api/v1/getcount")
# def get_count():
#     try:
#         count_of_embeddings = collector.count()
#         return {"count_of_embeddings": count_of_embeddings}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.delete("/api/v1/clear")
# def clear_data():
#     try:
#         collector.clear()
#         return {"message": "Data successfully cleared"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
# ---------------------------------------------------------------------------------------------------------------



@app.post("/api/v1/alloperations")
def all_operations(request: AllOpsRequest):
    '''
    Performs all operations in one go
    1. Add data
    2. Get data
    3. Delete data
    4. Clear data
    '''
    start_time = time.time()
    # TODO: Check if we can remove clear, this lock is needed for clear. If we are able to remove clear we can make this api server async.(but still this would require removing locks from the indiviidal DB Operations.)
    # Reason of using clear : Delete API does not free up storage.

    # lock is released only after all operations are completed, including delete() and clear().
    with collector.lock: 
        try:
            # Key Identifier 
            logfileurl = request.metadata.get('logfileurl')
            # Step 1: Add Data
            process_and_add_to_collector(request.corpus, collector, request.clear_before_adding, request.metadata)
            logger.info(f"Data successfully added for {logfileurl}")

            # Step 2: Get Data
            n_results = request.n_results or parameters.get_chunk_count() # "default": 250,
            max_token_count = request.max_token_count # or parameters.get_max_token_count()
            results = collector.get_sorted_by_dist(request.search_strings, n_results, max_token_count, metadata=request.metadata)
            logger.info(f"Data successfully fetched for {logfileurl}")
            return {"results": results}
        except Exception as e:
            logger.exception(f"Exception while processing {logfileurl} in process_and_add_to_collector: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            try:
                # Step 3 & 4: Delete and Clear Data
                collector.delete(ids_to_delete=None, where=request.metadata)
                collector.clear()  
                logger.info(f"Data successfully deleted and cleared for {logfileurl} | Time taken: {time.time() - start_time:.4f} seconds")
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error} for {logfileurl}")


@app.get('/')
def health():
    return {'msg': 'Welcome To ChromaDB!'}