"""
This module contains utils for preprocessing the text before converting it to embeddings.

- TextPreprocessorBuilder preprocesses individual strings.
    * stripping spaces
    * removing punctuation
    * lemmatizing
"""
import string
import nltk
import re
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
nltk.data.path.append(str(BASE_DIR / "nltk_data"))

from nltk.stem import WordNetLemmatizer


class TextPreprocessorBuilder:
     # Define class variables as None initially
    _lemmatizer = WordNetLemmatizer()
    
    # Some of the functions are expensive. We cache the results.
    _lemmatizer_cache = {}
    _pos_remove_cache = {}


    def __init__(self, text: str):
        self.text = text
    
    def strip(self):
        '''
        STEP: 7
        Remove leading and trailing spaces from the text.
        '''
        self.text = self.text.strip()
        return self
        
    def remove_punctuation(self):
        '''
        STEP: 2
        Remove all punctuation from the text.
        '''
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))
        return self

    def lemmatize(self):
        '''
        STEP: 5
        '''
        processed_text = TextPreprocessorBuilder._lemmatizer_cache.get(self.text)
        if processed_text:
            self.text = processed_text
            return self
        
        new_text = "".join([TextPreprocessorBuilder._lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b|\W+', self.text)])
        TextPreprocessorBuilder._lemmatizer_cache[self.text] = new_text
        self.text = new_text

        return self

    def build(self):
        '''
        STEP: 9
        '''
        return self.text