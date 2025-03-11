"""
This module provides a singleton class `Parameters` that is used to manage all hyperparameters for the embedding application.
It expects a JSON file in `app/config.json`.

Each element in the JSON must have a `default` value which will be used for the current run. Elements can have `categories`.
These categories define the range in which the optimizer will search. If the element is tagged with `"should_optimize": false`,
then the optimizer will only ever use the default value.
"""
import json
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

DIST_MIN_STRATEGY = 'Min of Two'
class Parameters:
    _instance = None

    variable_mapping = {
        'DIST_MIN_STRATEGY': DIST_MIN_STRATEGY
    }

    @staticmethod
    def getInstance():
        if Parameters._instance is None:
            Parameters()
        return Parameters._instance

    def __init__(self):
        if Parameters._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            logger.debug("Initializing Parameters singleton...")
            Parameters._instance = self
            BASE_DIR = Path(__file__).resolve().parent.parent
            self.hyperparameters = self._load_from_json(BASE_DIR / "config.json")

    def _load_from_json(self, file_path):
        logger.debug('Loading hyperparameters...')

        with open(file_path, 'r') as file:
            data = json.load(file)

        # Replace variable names in the dict and create Categorical objects
        for key in data:
            if "default" in data[key] and data[key]["default"] in self.variable_mapping:
                data[key]["default"] = self.variable_mapping[data[key]["default"]]
            if "categories" in data[key]:
                data[key]["categories"] = [self.variable_mapping.get(cat, cat) for cat in data[key]["categories"]]

        return data

def should_strip() -> bool:
    return bool(Parameters.getInstance().hyperparameters['strip']['default'])

def should_remove_punctuation() -> bool:
    return bool(Parameters.getInstance().hyperparameters['remove_punctuation']['default'])

def should_lemmatize() -> bool:
    return bool(Parameters.getInstance().hyperparameters['lemmatize']['default'])

def get_new_dist_strategy() -> str:
    return Parameters.getInstance().hyperparameters['new_dist_strategy']['default']

def get_chunk_count() -> int:
    return int(Parameters.getInstance().hyperparameters['chunk_count']['default'])

def get_significant_level() -> float:
    return float(Parameters.getInstance().hyperparameters['significant_level']['default'])

def get_max_token_count() -> int:
    return int(Parameters.getInstance().hyperparameters['max_token_count']['default'])