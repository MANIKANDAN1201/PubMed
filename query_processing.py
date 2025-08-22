from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from difflib import SequenceMatcher


# API Endpoints
MESH_API_BASE = "https://id.nlm.nih.gov/mesh/lookup"
DATAMUSE_API = "https://api.datamuse.com/words"
UMLS_API_BASE = "https://uts-ws.nlm.nih.gov/rest"


class EnhancedQueryProcessor:
    def __init__(self) -> None:
        self.medical_synonyms: Dict[str, List[str]] = {
            # Heart-related terms
            "heart attack": [
                "myocardial infarction",
                "cardiac arrest",
                "heart failure",
                "coronary thrombosis",
            ],
            "myocardial infarction": [
                "heart attack",
                "cardiac arrest",
                "coronary thrombosis",
                "heart failure",
            ],
            "cardiac arrest": [
                "heart attack",
                "myocardial infarction",
                "heart failure",
                "sudden cardiac death",
            ],

            # Air pollution terms
            "air pollution": [
                "air quality deterioration",
                "particulate matter",
                "PM2.5",
                "PM10",
                "airborne pollutants",
            ],
            "particulate matter": [
                "PM2.5",
                "PM10",
                "air pollution",
                "airborne particles",
                "fine particles",
            ],
            "PM2.5": ["particulate matter", "fine particles", "air pollution", "PM10"],
            "PM10": ["particulate matter", "coarse particles", "air pollution", "PM2.5"],

            # General medical terms
            "diabetes": [
                "diabetes mellitus",
                "type 1 diabetes",
                "type 2 diabetes",
                "blood sugar",
            ],
            "cancer": ["neoplasm", "malignancy", "tumor", "carcinoma"],
            "hypertension": ["high blood pressure", "HTN", "blood pressure"],
            "obesity": ["overweight", "body mass index", "BMI", "weight gain"],
        }

        self.mesh_mappings: Dict[str, str] = {
            "heart attack": "Myocardial Infarction",
            "myocardial infarction": "Myocardial Infarction",
            "cardiac arrest": "Heart Arrest",
            "air pollution": "Air Pollution",
            "particulate matter": "Particulate Matter",
            "PM2.5": "Particulate Matter",
            "PM10": "Particulate Matter",
            "diabetes": "Diabetes Mellitus",
            "cancer": "Neoplasms",
            "hypertension": "Hypertension",
            "obesity": "Obesity",
        }

    def _tokenize_and_group(self, text: str) -> List[str]:
        """Tokenize and group multi-word terms"""
        text = text.lower()
        # Keep medical terms together
        medical_terms = [
            "heart attack",
            "myocardial infarction",
            "cardiac arrest",
            "air pollution",
            "particulate matter",
            "blood pressure",
            "type 1 diabetes",
            "type 2 diabetes",
            "body mass index",
        ]

        # Replace medical terms with underscores to keep them as single tokens
        for term in medical_terms:
            if term in text:
                text = text.replace(term, term.replace(" ", "_"))

        # Tokenize
        tokens = re.split(r"\s+", text)
        tokens = [t for t in tokens if t and t not in ENGLISH_STOP_WORDS]

        # Restore medical terms
        restored_tokens = [t.replace("_", " ") for t in tokens]
        return restored_tokens

    def _get_general_synonyms(self, term: str) -> List[str]:
        """Get general synonyms using Datamuse API"""
        try:
            params = {"rel_syn": term, "max": 5}
            response = requests.get(DATAMUSE_API, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [item.get("word", "") for item in data[:3] if item.get("word")]
        except Exception:
            pass
        return []

    def _get_medical_synonyms(self, term: str) -> List[str]:
        """Get medical synonyms from our curated database"""
        term_lower = term.lower()
        return self.medical_synonyms.get(term_lower, [])

    def _get_mesh_term(self, term: str) -> Optional[str]:
        """Get official MeSH term mapping"""
        term_lower = term.lower()
        return self.mesh_mappings.get(term_lower)

    def _mesh_lookup_terms(self, label: str, limit: int = 10) -> List[Dict]:
        """Lookup MeSH terms via NCBI API"""
        try:
            params = {"label": label, "match": "approximate", "limit": str(limit)}
            url = f"{MESH_API_BASE}/term"
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    def _collect_mesh_synonyms(self, label: str) -> List[str]:
        """Collect MeSH synonyms and related terms"""
        try:
            candidates = self._mesh_lookup_terms(label)
            syns: List[str] = []
            for c in candidates:
                lbl = c.get("label") or ""
                if lbl:
                    syns.append(lbl)
                term = c.get("term") or ""
                if term and term.lower() != lbl.lower():
                    syns.append(term)
            # Deduplicate and normalize
            return sorted(list({s.lower() for s in syns if s}))
        except Exception:
            return []

    def _similarity_score(self, a: str, b: str) -> float:
        """Calculate string similarity"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def expand_term_synonyms(self, term: str) -> List[str]:
        """Comprehensive synonym expansion for a term"""
        synonyms: set[str] = set()

        # 1. Add original term
        synonyms.add(term.lower())

        # 2. Medical synonyms from curated database
        medical_syns = self._get_medical_synonyms(term)
        synonyms.update(medical_syns)

        # 3. General synonyms from Datamuse
        general_syns = self._get_general_synonyms(term)
        synonyms.update(general_syns)

        # 4. MeSH synonyms
        mesh_syns = self._collect_mesh_synonyms(term)
        synonyms.update(mesh_syns)

        # 5. Check for similar terms in our database
        for known_term, known_syns in self.medical_synonyms.items():
            if self._similarity_score(term, known_term) > 0.8:
                synonyms.update(known_syns)

        return sorted(list(synonyms))

    def build_enhanced_boolean_query(self, term_groups: List[List[str]]) -> str:
        """Build PubMed-optimized boolean query"""
        query_parts: List[str] = []

        for group in term_groups:
            if not group:
                continue

            # Get MeSH term for the primary term
            primary_term = group[0]
            mesh_term = self._get_mesh_term(primary_term)

            # Build OR group
            or_terms: List[str] = []

            # Add MeSH term if available
            if mesh_term:
                or_terms.append(f'"{mesh_term}"[MeSH Terms]')

            # Add all synonyms as [All Fields]
            for term in group:
                if " " in term:
                    or_terms.append(f'"{term}"[All Fields]')
                else:
                    or_terms.append(f"{term}[All Fields]")

            if or_terms:
                query_parts.append(f"({' OR '.join(or_terms)})")

        return " AND ".join(query_parts)

    def process_query(self, query: str) -> Tuple[str, Dict[str, List[str]], List[str]]:
        """Enhanced query processing pipeline"""
        # 1. Tokenize and group terms
        tokens = self._tokenize_and_group(query)

        # 2. Expand synonyms for each term
        expanded_groups: List[List[str]] = []
        synonyms_map: Dict[str, List[str]] = {}

        for token in tokens:
            synonyms = self.expand_term_synonyms(token)
            expanded_groups.append(synonyms)
            synonyms_map[token] = synonyms

        # 3. Build enhanced boolean query
        enhanced_query = self.build_enhanced_boolean_query(expanded_groups)

        return enhanced_query, synonyms_map, tokens


# Global processor instance
_processor = EnhancedQueryProcessor()


def expand_query(query: str, email: str = "") -> Tuple[str, Dict[str, List[str]], List[str]]:
    """
    Enhanced query expansion with medical synonyms, MeSH mapping, and boolean structuring
    Returns (enhanced_boolean_query, synonyms_map, filtered_tokens)
    """
    return _processor.process_query(query)


