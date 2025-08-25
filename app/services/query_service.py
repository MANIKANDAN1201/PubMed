from typing import Tuple, Dict, List
from app.services.logic.query_processing import expand_query


def process_query(query: str, email: str = "") -> Tuple[str, Dict[str, List[str]], List[str]]:
    return expand_query(query, email)


