"""
Data Export utilities for PubMed Semantic Search
Contains functions for exporting search results to various formats.
"""

import pandas as pd
from typing import List, Dict, Any

def prepare_results_for_export(sorted_results: List[Dict], query: str) -> pd.DataFrame:
    """Prepare search results for CSV export"""
    if not sorted_results:
        return pd.DataFrame()
    
    selected = []
    for rank, result in enumerate(sorted_results, start=1):
        art = result["art"]
        score = result["score"]
        meta = result["meta"]
        selected.append({
            "rank": rank,
            "pmid": art.pmid,
            "title": art.title,
            "journal": art.journal or "",
            "year": art.year or "",
            "url": art.url,
            "final_score": float(score),
            "semantic_score": getattr(meta, 'semantic_score', 0),
            "keyword_score": getattr(meta, 'keyword_score', 0),
            "abstract": art.abstract,
        })
    
    return pd.DataFrame(selected)

def generate_csv_filename(query: str) -> str:
    """Generate a clean filename for CSV export"""
    clean_query = query.replace(' ', '_')[:30]
    return f"pubmed_enhanced_results_{clean_query}.csv"
