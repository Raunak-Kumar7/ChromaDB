from pydantic import BaseModel
from typing import Dict, List, Optional

class AddRequest(BaseModel):
    corpus: str
    clear_before_adding: Optional[bool] = False
    metadata: Optional[Dict] = None

class DeleteRequest(BaseModel):
    metadata: Dict

class GetRequest(BaseModel):
    search_strings: List[str]
    n_results: Optional[int] = None
    max_token_count: Optional[int] = None
    metadata: Optional[Dict] = {}


class AllOpsRequest(BaseModel):
    corpus: str
    clear_before_adding: Optional[bool] = False
    metadata: Optional[Dict] = None
    search_strings: List[str]
    n_results: Optional[int] = None
    max_token_count: Optional[int] = None