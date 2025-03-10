"""
This module is responsible for processing the corpus and feeding it into chromaDB. It will receive a corpus of text. 
It will then split it into chunks of specified length. For each of those chunks, it will append surrounding context.
It will only include full words.
"""

import re
import bisect

import app.utils.parameters as parameters

from app.utils.data_preprocessor import TextPreprocessorBuilder#, TextSummarizer
from app.chromadb.chromaClient import ChromaCollector
import logging

logger = logging.getLogger(__name__)

def preprocess_text_no_summary(text) -> str:
    '''
    Uses the TextPreprocessorBuilder to preprocess the text passed.
    '''
    builder = TextPreprocessorBuilder(text)

    # Default remove Punctuation -> true
    if parameters.should_remove_punctuation():
        builder.remove_punctuation()

    # Default Lemmatize -> true
    if parameters.should_lemmatize():
        builder.lemmatize()

    # Default strip -> true
    if parameters.should_strip():
        builder.strip()
    
    return builder.build()

def _create_chunks_with_context(corpus, chunk_len, context_left, context_right):
    """
    This function takes a corpus of text and splits it into chunks of a specified length, 
    then adds a specified amount of context to each chunk. The context is added by first 
    going backwards from the start of the chunk and then going forwards from the end of the 
    chunk, ensuring that the context includes only whole words and that the total context length 
    does not exceed the specified limit. This function uses binary search for efficiency.

    Returns:
    chunks (list of str): The chunks of text.
    chunks_with_context (list of str): The chunks of text with added context.
    chunk_with_context_start_indices (list of int): The starting indices of each chunk with context in the corpus.
    """
    # STEP 1: Splits the corpus into words and spaces while preserving the spaces. Eg. words = ["Hello", " ", "world!", " ", "How", " ", "are", " ", "you?"]
    words = re.split('(\\s+)', corpus)
    
    # STEP 2: Creating a List of Word Start Indices. This keeps track of where each word starts in the original corpus. eg. word_start_indices = [0, 5, 6, 12, 13, 16, 17, 20, 21, 25]
    word_start_indices = [0]
    current_index = 0
    for word in words:
        current_index += len(word)
        word_start_indices.append(current_index)

    # STEP 3: Creating Chunks Without Context
    # 1. List of text chunks, 2. Length of each chunk. 3. Start positions of each chunk.
    chunks, chunk_lengths, chunk_start_indices  = [], [], []
    current_length = 0 # To track length of current chunk
    current_index = 0
    chunk = []

    for word in words:
        if current_length + len(word) > chunk_len:
            chunks.append(''.join(chunk)) # append a new chunk in the list
            chunk_lengths.append(current_length) # append the length of the chunk
            chunk_start_indices.append(current_index - current_length) # append the starting index of the chunk
            
            # Start a new chunk with the current word
            chunk = [word]
            current_length = len(word)
        else:
            chunk.append(word)
            current_length += len(word)
        current_index += len(word)
    
    # Add the last chunk (CASE: if it did not exceed the chunk length)
    if chunk:
        chunks.append(''.join(chunk))
        chunk_lengths.append(current_length)
        chunk_start_indices.append(current_index - current_length)

    # STEP 4: Creating Chunks With Context
    # 1. List of chunks with Context, 2. Start positions of each chunk with context
    chunks_with_context, chunk_with_context_start_indices = [], []

    for start_index, chunk_length in zip(chunk_start_indices, chunk_lengths):

        # Instead of simply adding context_left or context_right as a fixed number of characters, 
        # the code is using binary search (via bisect) to find the correct word boundaries. word_start_indices:  list contains the starting indices of each word in the corpus.
        context_start_index = bisect.bisect_right(word_start_indices, start_index - context_left)
        context_end_index = bisect.bisect_left(word_start_indices, start_index + chunk_length + context_right)

        # Combine all the words in the context range (before, chunk, and after)
        chunk_with_context = ''.join(words[context_start_index:context_end_index])
        chunks_with_context.append(chunk_with_context)
        
        # Determine the start index of the chunk with context
        chunk_with_context_start_index = word_start_indices[context_start_index]
        chunk_with_context_start_indices.append(chunk_with_context_start_index)

    return chunks, chunks_with_context, chunk_with_context_start_indices


def _clear_chunks(data_chunks, data_chunks_with_context, data_chunk_starting_indices):
    '''
    Step 1: Skips chunks that are empty or contain only symbols/spaces.
    Step 2: Skip chunks if we've already seen this exact chunk, and the context around it it very close to the seen chunk. (TODO: Currently not doing it.)
    '''
    # 1. Chunks 2. Chunks + left Context + right Context 3. Starting index of each Chunk
    distinct_data_chunks = []
    distinct_data_chunks_with_context = []
    distinct_data_chunk_starting_indices = []

    # Stores unique chunks.
    seen_chunks = dict()

    for chunk, context, index in zip(data_chunks, data_chunks_with_context, data_chunk_starting_indices):

        # Skip the chunk if it does not contain any alphanumeric characters
        # Skips chunks that are empty or contain only symbols/spaces.
        if not any(char.isalnum() for char in chunk):
            continue

        distinct_data_chunks.append(chunk)
        distinct_data_chunks_with_context.append(context)
        distinct_data_chunk_starting_indices.append(index)

        # Stores the first occurrence index of this chunk in seen_chunks.
        seen_chunks[chunk] = index

    return distinct_data_chunks, distinct_data_chunks_with_context, distinct_data_chunk_starting_indices


# You will get a collector, your work is just to process the corpus and add to the given collector
def process_and_add_to_collector(corpus: str, collector: ChromaCollector, clear_collector_before_adding: bool, metadata: dict):
    '''
    STEP 1: Define Chunk Size & Context Size
    Chunk Size: Defines how much text is in a single chunk. If chunk_len = 700, each chunk will have around 700 characters.
    Context Size: Adds extra words before and after a chunk to provide better understanding.
        - context_left: Number of extra characters/words before the chunk. 250
		- context_right: Number of extra characters/words after the chunk. 800
    
    STEP 2: Create 1. Chunks 2. Chunks + left Context + right Context 3. Starting index of each Chunk
    
    STEP 3: Preprocess the text in each chunk

    STEP 4: Remove unwanted chunks if empty or duplicate

    STEP 5: Add the data to the collector
    '''
    # TODO: Experiment to have multiple Chunk Sizes in list, this could help in having related data together in some chunks. Might slow down the Search as well.
    chunk_lens = [700]
    context_left = 250
    context_right = 800


    # 1. Chunks 2. Chunks + left Context + right Context 3. Starting index of each Chunk
    data_chunks = []
    data_chunks_with_context = []
    data_chunk_starting_indices = []
    
    # Fixed Size Chunking 
    for chunk_len in chunk_lens:
        chunks, chunks_with_context, context_start_indices = _create_chunks_with_context(corpus, chunk_len, context_left, context_right)
        data_chunks.extend(chunks)
        data_chunks_with_context.extend(chunks_with_context)
        data_chunk_starting_indices.extend(context_start_indices)

    data_chunks = [preprocess_text_no_summary(chunk) for chunk in data_chunks]

    # Remove unwanted chunks if empty or duplicate
    data_chunks, data_chunks_with_context, data_chunk_starting_indices = _clear_chunks(
        data_chunks, data_chunks_with_context, data_chunk_starting_indices
    )

    if clear_collector_before_adding:
        collector.clear()
    # Calls the collector and adds the data along with metadata
    collector.add(data_chunks, data_chunks_with_context, data_chunk_starting_indices, [metadata]*len(data_chunks) if metadata is not None else None)
