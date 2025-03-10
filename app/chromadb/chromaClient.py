import math
import threading

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from app.utils.util import encode, decode
from collections import OrderedDict
import app.utils.parameters as parameters
from app.logging_config import setup_logging
import logging


setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting the application.")

embedder = embedding_functions.SentenceTransformerEmbeddingFunction("sentence-transformers/all-mpnet-base-v2")


class Info:
    '''
    Data Structure used to store and manipulate Chunks with Context.
    Class only used while Retrieving data from ChromaDB
    '''

    def __init__(self, start_index, text_with_context, distance, id):
        '''
        Params:
            start_index: Start Index of Chunk in full doc
            text_with_context: Chunk with some left and right context.
            distance: Distance of the chunk from the search string
            id: Id given to that chunk while adding data to DB
        '''
        self.text_with_context = text_with_context
        self.start_index = start_index
        self.distance = distance
        self.id = id

    def calculate_distance(self, other_info):
        '''
        Compares distances between two Info objects.
        Returns the minimum distance from Search String between the two Info objects.
        '''
        if parameters.get_new_dist_strategy() == parameters.DIST_MIN_STRATEGY:
            return min(self.distance, other_info.distance)

    def merge_with(self, other_info):
        '''
        Merges two Info objects if they are adjacent or overlapping.
        Merged text retains the 
            - smaller start_index 
            - shorter similarity distance.
            - text_with_context is the combination of the two texts.(handles overlaps)
            - id is the id of the Info object with the smaller start_index.
        '''
        s1 = self.text_with_context
        s2 = other_info.text_with_context
        s1_start = self.start_index
        s2_start = other_info.start_index

        new_dist = self.calculate_distance(other_info)

        if self.should_merge(s1, s2, s1_start, s2_start):
            if s1_start <= s2_start:
                if s1_start + len(s1) >= s2_start + len(s2):  # if s1 completely covers s2
                    return Info(s1_start, s1, new_dist, self.id)
                else:
                    overlap = max(0, s1_start + len(s1) - s2_start)
                    return Info(s1_start, s1 + s2[overlap:], new_dist, self.id)
            else:
                if s2_start + len(s2) >= s1_start + len(s1):  # if s2 completely covers s1
                    return Info(s2_start, s2, new_dist, other_info.id)
                else:
                    overlap = max(0, s2_start + len(s2) - s1_start)
                    return Info(s2_start, s2 + s1[overlap:], new_dist, other_info.id)

        return None

    @staticmethod
    def should_merge(s1, s2, s1_start, s2_start):
        '''
        Checks if two texts should be merged.
        If they are adjacent or overlapping, returns True.
        '''

        # Check if s1 and s2 are adjacent or overlapping
        s1_end = s1_start + len(s1)
        s2_end = s2_start + len(s2)

        # if they overlap, it Returns True
        return not (s1_end < s2_start or s2_end < s1_start)

class ChromaCollector():
    '''
    ChromaCollector is a class which handles all the ChromaDB operations.
    ChromaDB is running as a in memory DB via chromadb python package.
    '''

    def __init__(self) -> None:

        # Can configure the settings to change the client type https://cookbook.chromadb.dev/core/clients/#chroma-clients
        self.chroma_client = chromadb.Client(settings=Settings(allow_reset=True, anonymized_telemetry=False))


        # Create A new Collection (1 collection per client)
        self.collection_name = "context"
        self.collection = self.chroma_client.create_collection(name=self.collection_name, embedding_function=embedder)

        # Stores Unique incremental ids, 1 for each chunk
        self.ids = []
        self.id_to_info = {}
        self.max_cache_size = 10000  
        self.embeddings_cache = OrderedDict()  # FIFO cache

        # Locking so the server doesn't break.
        # lock (self.lock) is shared across all the methods.
        # If multiple API calls are handled by different threads within the same server process, the self.lock will ensure that only one API request at a time enters the critical section.

        self.lock = threading.Lock()

    def _get_new_ids(self, num_new_ids: int):
        '''
        Param:
            num_new_ids: No. of new ids need to be generated
        To generate new unique ID for each Chunk stored in ChromaDB
        '''
        # STEP 1: Find out the Highest ID stored in self.ids
        if self.ids:
            max_existing_id = max(int(id_) for id_ in self.ids)
        else:
            max_existing_id = -1

        # STEP 2: Creates and returns a list of new unique IDs
        return [str(i + max_existing_id + 1) for i in range(num_new_ids)]

    def _split_texts_by_cache_hit(self, texts: list[str], new_ids: list[str], metadatas: list[dict]):
        '''
        Params:
            texts: List of Chunks
            new_ids: List of Unique IDs for each Chunk
            metadatas: List of SAME metadata for each of the chunk.
        Separates cached and non-cached texts.
        Returns two tuples:
            - Tuple 1: (existing_texts, existing_embeddings, existing_ids, existing_metas)
            - Tuple 2: (non_existing_texts, non_existing_ids, non_existing_metas)
        '''
        existing_texts, non_existing_texts = [], []
        existing_embeddings = []
        existing_ids, non_existing_ids = [], []
        existing_metas, non_existing_metas = [], []

        for i, text in enumerate(texts):
            id_ = new_ids[i]
            metadata = metadatas[i] if metadatas is not None else None
            embedding = self.embeddings_cache.get(text)  # Checks if text is in cache.
            if embedding is not None:
                existing_texts.append(text)
                existing_embeddings.append(embedding)
                existing_ids.append(id_)
                existing_metas.append(metadata)
            else:
                non_existing_texts.append(text)
                non_existing_ids.append(id_)
                non_existing_metas.append(metadata)

        return (existing_texts, existing_embeddings, existing_ids, existing_metas), \
               (non_existing_texts, non_existing_ids, non_existing_metas)

    # Main function to add documents to the collection
    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int], metadatas: list[dict] = None):
        '''
        Params:
            text: list of Chunks broken by data_processor.py
            text_with_context: list of same Chunks with some left and right context.
            starting_indices: list of starting index of each respective chunk
            metadata: list of SAME metadata for each of the chunk.
        Adds the chunks to the ChromaDB collection.
        '''

        # Ensure that if metadata is provided, it matches the number of text chunks or is NONE
        if metadatas is not None and len(metadatas) != len(texts):
            logger.error(f"Metadata length mismatch! metadatas={len(metadatas)}, texts={len(texts)}")
            raise ValueError("metadatas must be None or have the same length as texts")

        # Exit if no chunks passed.
        if len(texts) == 0:
            logger.warning("No texts provided for addition. Skipping add operation.")
            return
        
        # Generate list of unique ID for each Chunk.
        new_ids = self._get_new_ids(len(texts))
        
        # Separates already cached and non-cached texts.
        (existing_texts, existing_embeddings, existing_ids, existing_metas), \
            (non_existing_texts, non_existing_ids, non_existing_metas) = self._split_texts_by_cache_hit(texts, new_ids, metadatas)
                
        # STEP: Add Already Cached Embeddings [all at once]
        # If there are any already existing texts, add them all at once.
        if existing_texts:
            logger.info(f'Adding {len(existing_embeddings)} cached embeddings.')
            args = {'embeddings': existing_embeddings, 'documents': existing_texts, 'ids': existing_ids}
            if metadatas is not None:
                args['metadatas'] = existing_metas
            self.collection.add(**args)

        # STEP: Add Non-Cached Embeddings [all at once]
        # If there are any non-existing texts, compute their embeddings all at once. Each call to embed has significant overhead.
        if non_existing_texts:
            non_existing_embeddings = embedder(non_existing_texts) # batch-processes non_existing_texts, which takes a list of texts and returns a list of corresponding embeddings.
            for text, embedding in zip(non_existing_texts, non_existing_embeddings):
                if len(self.embeddings_cache) >= self.max_cache_size:
                    self.embeddings_cache.popitem(last=False) # O(1) operation, Keeps the Cache from growing infinetly
                self.embeddings_cache[text] = embedding # Stores new embeddings in self.embeddings_cache to avoid recomputation in the future.

            logger.info(f'Adding {len(non_existing_embeddings)} new embeddings.')
            args = {'embeddings': non_existing_embeddings, 'documents': non_existing_texts, 'ids': non_existing_ids}
            if metadatas is not None:
                args['metadatas'] = non_existing_metas
            self.collection.add(**args)

        # STEP: Create a dictionary that maps each ID to its context and starting index
        new_info = {
            id_: {'text_with_context': context, 'start_index': start_index}
            for id_, context, start_index in zip(new_ids, texts_with_context, starting_indices)
        }

        self.id_to_info.update(new_info) # Updates the Dict, prevents duplicate keys.
        self.ids.extend(new_ids)
        self.embeddings_cache.clear() # Data not shared accross api calls, so clearing it.


    def _filter_outliers_by_median_distance(self, infos: list[Info], significant_level: float):
        '''
        Params:
            - infos: List of Info objects
            - significant_level: Level of significance to filter out the outliers. "default": 1.0
        
        Removes info that have a distance greater than the median by a significant level.
        Returns a list of filtered infos.
        '''
        # Ensure there are infos to filter
        if not infos:
            return []

        # STEP1: Find info with minimum distance. This ensures that at least one valid entry is always kept in the results.
        min_info = min(infos, key=lambda x: x.distance)

        # STEP2: Calculate median distance among infos
        median_distance = np.median([inf.distance for inf in infos])

        # Filter out infos that have a distance significantly greater than the median
        filtered_infos = [inf for inf in infos if inf.distance <= significant_level * median_distance]

        # Always include the info with minimum distance
        if min_info not in filtered_infos:
            filtered_infos.append(min_info)

        return filtered_infos

    def _merge_infos(self, infos: list[Info]):
        '''
        Params:
            - infos: List of Info objects which are already sorted by Start Index
        
        Merges a list of Info objects if they are adjacent or overlapping. 
        The goal is to reduce redundancy and create a more compact representation of text snippets.
        Returns: a list of Infos that have been merged.
        '''
        merged_infos = []
        current_info = infos[0]

        for next_info in infos[1:]:
            merged = current_info.merge_with(next_info)
            if merged is not None:
                current_info = merged
            else:
                merged_infos.append(current_info)
                current_info = next_info

        merged_infos.append(current_info)
        return merged_infos


    def _get_documents_ids_distances(self, search_strings: list[str], n_results: int, metadata: dict = None):
        '''
        Params:
            - search_strings: List of search strings.
            - n_results: Number of Chunks to return.
            - metadata: Metadata to filter the search results.

        Returns : List of Most similar upto n_results Chunks with Context, IDs, and Distances of the Chunks.
        '''
        # STEP: Limit the number of results to the number of ids in the collection
        n_results = min(len(self.ids), n_results)
        if n_results == 0:
            return [], [], []

        # STEP: If search_string is passed as a String, Convert it to a single object list.
        if isinstance(search_strings, str):
            search_strings = [search_strings]

        infos = []

        for search_string in search_strings:

            query_params = {"query_texts": [search_string],                           # Searches for ith search_string in the document collection.
                            "n_results": math.ceil(n_results / len(search_strings)),  # Distributes n_results evenly across all search strings (n_results / len(search_strings)).
                            "include": ['distances']}                                 # Retrieves distance scores (similarity scores).
            if metadata:
                query_params["where"] = metadata                                      # Adds metadata conditions to filter results.
            result = self.collection.query(**query_params)
            curr_infos = [Info(start_index=self.id_to_info[id]['start_index'],
                               text_with_context=self.id_to_info[id]['text_with_context'],
                               distance=distance, id=id)
                          for id, distance in zip(result['ids'][0], result['distances'][0])] #  Extracts results and converts them into Info objects.

            curr_infos = self._filter_outliers_by_median_distance(curr_infos, parameters.get_significant_level())   # Removes info that have a distance greater than the median by a significant level.
            infos.extend(curr_infos)

        # STEP: Sortinng List of Infos by start_index, So that we only need to merge the adjacent ones, If possible.
        infos.sort(key=lambda x: x.start_index)
        infos = self._merge_infos(infos)

        # STEP: Return the text_with_context, ids, and distances of the Final Merged Unique Infos
        texts_with_context = [inf.text_with_context for inf in infos]
        ids = [inf.id for inf in infos]
        distances = [inf.distance for inf in infos]

        return texts_with_context, ids, distances

    def _get_documents_up_to_token_count(self, documents: list[str], max_token_count: int):
        '''
        Params:
            documents: list of documents
            max_token_count: Maximum token count allowed for retrieval.
            TODO: Check any alternative to encode() and decode() functions here using tiktoken library.
        Returns a list of documents in the same order that have a total token count less than or equal to max_token_count.
        '''
        current_token_count = 0
        return_documents = []

        for doc in documents:
            # tokenizes each document to count how many tokens it has
            doc_tokens = encode(doc)  # Use the new encode function from utils.py
            doc_token_count = len(doc_tokens)
            
            if current_token_count + doc_token_count > max_token_count:
                remaining_tokens = max_token_count - current_token_count
                
                # decode() to convert the truncated tokens back into a string and adds it to return_documents
                truncated_doc = decode(doc_tokens[:remaining_tokens])  # Use the new decode function from utils.py
                return_documents.append(truncated_doc)
                break
            else:
                return_documents.append(doc)
                current_token_count += doc_token_count
        return return_documents

    def get_sorted_by_dist(self, search_strings: list[str], n_results: int, max_token_count: int, metadata: dict = None) -> list[str]:
        '''
        Params:
            - search_strings: List of search strings.
            - n_results: Number of Chunks to return. "default": 250
            - max_token_count: Maximum token count allowed for retrieval.
            - metadata: Metadata to filter the search results.
        Main function to get chunks by similarity and then sort by distance (lowest distance is last).
        Retrieves, sorts, truncates, and returns documents based on their distance scores while ensuring they fit within a given token limit.

        '''
        # Retrieves documents matching search_strings and their corresponding distances, Uses n_results to limit how many chunks to retrieve.
        documents, _, distances = self._get_documents_ids_distances(search_strings, n_results, metadata=metadata)
        # Sorts them in ascending order (closest match first).
        sorted_docs = [doc for doc, _ in sorted(zip(documents, distances), key=lambda x: x[1])]  # sorted lowest -> highest

        # return sorted_docs # to prevent tokenizer/model requried for the superboogav2 get Context.

        # list of documents in the same order that have a total token count less than or equal to max_token_count.
        # If a document is truncated or competely skipped, it would be with high distance.
        return_documents = self._get_documents_up_to_token_count(sorted_docs, max_token_count)

        # reversing helps by making sure the most similar details are last.
        return_documents.reverse()  # highest -> lowest

        return return_documents

    def delete(self, ids_to_delete: list[str], where: dict):
        '''
        Deleting records based on where(metadata) solely
        '''
        ids_to_delete = self.collection.get(ids=ids_to_delete, where=where)['ids'] # Fetches ids based on metadata
        self.collection.delete(ids=ids_to_delete, where=where)
        # Remove the deleted ids from self.ids and self.id_to_info
        ids_set = set(ids_to_delete)
        self.ids = [id_ for id_ in self.ids if id_ not in ids_set]
        for id_ in ids_to_delete:
            self.id_to_info.pop(id_, None)

        logger.info(f'Successfully deleted {len(ids_to_delete)} records from chromaDB.')

    def clear(self):
        '''
        Resets the ChromaDB clients, and creates a new Collection
        Resets ids, id_to_info
        '''
        self.chroma_client.reset()
        self.collection = self.chroma_client.create_collection("context", embedding_function=embedder)
        self.ids = []
        self.id_to_info = {}

        logger.info('Successfully cleared all records and reset chromaDB.')

    def count(self):
        '''
        Returns count of embedding in the collection
        '''
        count_of_embeddings = self.collection.count()
        logger.info(f'Number of Embeddings in the collection {count_of_embeddings}')
        return count_of_embeddings
        

def make_collector():
    return ChromaCollector()