from typing import Tuple, Dict, List
try:
    from .logic.query_processing import expand_query
except Exception:
    from ...query_processing import expand_query


def process_query(query: str, email: str = "") -> Tuple[str, Dict[str, List[str]], List[str]]:
    return expand_query(query, email)


