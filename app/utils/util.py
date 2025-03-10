'''
TODO: Find if tiktoken gradually increases the memory usage.
Or if we can use any other way/lib to count the number of tokens.
'''
import tiktoken
import logging
from app.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Loading Encoding Model...")
encoding = tiktoken.get_encoding("cl100k_base")

def encode(text: str) -> list[int]:
    """Encodes the given text into a list of tokens using the specified tokenizer."""
    # Encode the text
    encoded_text = encoding.encode(text)

    return encoded_text

def decode(tokens: list[int]) -> str:
    """Decodes the given token list back into text."""

    # Decode the tokens
    decoded_text = encoding.decode(tokens)

    return decoded_text