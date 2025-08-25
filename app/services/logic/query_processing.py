from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from difflib import SequenceMatcher


MESH_API_BASE = "https://id.nlm.nih.gov/mesh/lookup"
DATAMUSE_API = "https://api.datamuse.com/words"
UMLS_API_BASE = "https://uts-ws.nlm.nih.gov/rest"


class EnhancedQueryProcessor:
    def __init__(self) -> None:
        self.medical_synonyms: Dict[str, List[str]] = {
            "heart attack": ["myocardial infarction", "cardiac arrest", "heart failure", "coronary thrombosis"],
            "myocardial infarction": ["heart attack", "cardiac arrest", "coronary thrombosis", "heart failure"],
            "cardiac arrest": ["heart attack", "myocardial infarction", "heart failure", "sudden cardiac death"],
            "air pollution": ["air quality deterioration", "particulate matter", "PM2.5", "PM10", "airborne pollutants"],
            "particulate matter": ["PM2.5", "PM10", "air pollution", "airborne particles", "fine particles"],
            "PM2.5": ["particulate matter", "fine particles", "air pollution", "PM10"],
            "PM10": ["particulate matter", "coarse particles", "air pollution", "PM2.5"],
            "diabetes": ["diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "blood sugar"],
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
        text = text.lower()
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
        for term in medical_terms:
            if term in text:
                text = text.replace(term, term.replace(" ", "_"))
        tokens = re.split(r"\s+", text)
        tokens = [t for t in tokens if t and t not in ENGLISH_STOP_WORDS]
        restored_tokens = [t.replace("_", " ") for t in tokens]
        return restored_tokens

    def _get_general_synonyms(self, term: str) -> List[str]:
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
        term_lower = term.lower()
        return self.medical_synonyms.get(term_lower, [])

    def _get_mesh_term(self, term: str) -> Optional[str]:
        term_lower = term.lower()
        return self.mesh_mappings.get(term_lower)

    def _mesh_lookup_terms(self, label: str, limit: int = 10) -> List[Dict]:
        try:
            params = {"label": label, "match": "approximate", "limit": str(limit)}
            url = f"{MESH_API_BASE}/term"
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    def _collect_mesh_synonyms(self, label: str) -> List[str]:
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
            return sorted(list({s.lower() for s in syns if s}))
        except Exception:
            return []

    def _similarity_score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def expand_term_synonyms(self, term: str) -> List[str]:
        synonyms: set[str] = set()
        synonyms.add(term.lower())
        synonyms.update(self._get_medical_synonyms(term))
        synonyms.update(self._get_general_synonyms(term))
        synonyms.update(self._collect_mesh_synonyms(term))
        for known_term, known_syns in self.medical_synonyms.items():
            if self._similarity_score(term, known_term) > 0.8:
                synonyms.update(known_syns)
        return sorted(list(synonyms))

    def build_enhanced_boolean_query(self, term_groups: List[List[str]]) -> str:
        query_parts: List[str] = []
        for group in term_groups:
            if not group:
                continue
            primary_term = group[0]
            mesh_term = self._get_mesh_term(primary_term)
            or_terms: List[str] = []
            if mesh_term:
                or_terms.append(f'"{mesh_term}"[MeSH Terms]')
            for term in group:
                if " " in term:
                    or_terms.append(f'"{term}"[All Fields]')
                else:
                    or_terms.append(f"{term}[All Fields]")
            if or_terms:
                query_parts.append(f"({' OR '.join(or_terms)})")
        return " AND ".join(query_parts)

    def process_query(self, query: str) -> Tuple[str, Dict[str, List[str]], List[str]]:
        tokens = self._tokenize_and_group(query)
        expanded_groups: List[List[str]] = []
        synonyms_map: Dict[str, List[str]] = {}
        for token in tokens:
            synonyms = self.expand_term_synonyms(token)
            expanded_groups.append(synonyms)
            synonyms_map[token] = synonyms
        enhanced_query = self.build_enhanced_boolean_query(expanded_groups)
        return enhanced_query, synonyms_map, tokens


_processor = EnhancedQueryProcessor()


def expand_query(query: str, email: str = "") -> Tuple[str, Dict[str, List[str]], List[str]]:
    return _processor.process_query(query)


