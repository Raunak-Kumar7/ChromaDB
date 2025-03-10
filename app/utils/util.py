'''
TODO: Find if tiktoken gradually increases the memory usage.
Or if we can use any other way/lib to count the number of tokens.
'''
import tiktoken

def encode(text: str) -> list[int]:
    """Encodes the given text into a list of tokens using the specified tokenizer."""
    # Get encoding for a specific model
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text
    encoded_text = encoding.encode(text)

    return encoded_text

def decode(tokens: list[int]) -> str:
    """Decodes the given token list back into text."""
    # Get encoding for a specific model
    encoding = tiktoken.get_encoding("cl100k_base")

    # Decode the tokens
    decoded_text = encoding.decode(tokens)

    return decoded_text