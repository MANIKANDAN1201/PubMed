from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass
from Bio import Entrez
import requests


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str
    journal: Optional[str] = None
    year: Optional[str] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    is_free: bool = False
    full_text_link: Optional[str] = None


def _extract_doi(entry: dict) -> Optional[str]:
    """Best-effort extraction of DOI from PubMed XML entry."""
    try:
        medline = entry.get("MedlineCitation", {})
        article = medline.get("Article", {})

        # Option 1: ELocationID with EIdType = doi
        eids = article.get("ELocationID")
        if isinstance(eids, list):
            for eid in eids:
                if isinstance(eid, dict) and eid.get("@EIdType", "").lower() == "doi":
                    val = eid.get("#text")
                    if val:
                        return str(val).strip()
        elif isinstance(eids, dict):
            if eids.get("@EIdType", "").lower() == "doi":
                val = eids.get("#text")
                if val:
                    return str(val).strip()

        # Option 2: PubmedData -> ArticleIdList -> ArticleId with IdType = doi
        pubmed_data = entry.get("PubmedData", {})
        id_list = pubmed_data.get("ArticleIdList") or []
        if isinstance(id_list, list):
            for aid in id_list:
                if isinstance(aid, dict) and aid.get("@IdType", "").lower() == "doi":
                    val = aid.get("#text")
                    if val:
                        return str(val).strip()
        elif isinstance(id_list, dict):
            # sometimes a single dict
            if id_list.get("@IdType", "").lower() == "doi":
                val = id_list.get("#text")
                if val:
                    return str(val).strip()
    except Exception:
        pass
    return None


def _parse_article(entry: dict) -> Optional[PubMedArticle]:
    try:
        medline = entry.get("MedlineCitation", {})
        
        # Handle PMID - can be direct value or dict with #text
        pmid_raw = medline.get("PMID")
        if isinstance(pmid_raw, dict):
            pmid = pmid_raw.get("#text")
        else:
            pmid = pmid_raw
            
        if not pmid:
            return None
            
        article = medline.get("Article", {})
        title = article.get("ArticleTitle") or ""
        
        # Handle abstract - can be list of StringElements or direct string
        abstract_parts = article.get("Abstract", {}).get("AbstractText")
        abstract = ""
        
        if isinstance(abstract_parts, list):
            # Extract text from StringElements
            abstract_texts = []
            for part in abstract_parts:
                if hasattr(part, 'content'):  # StringElement
                    abstract_texts.append(str(part.content))
                else:
                    abstract_texts.append(str(part))
            abstract = " ".join(abstract_texts)
        elif isinstance(abstract_parts, str):
            abstract = abstract_parts
        elif abstract_parts:
            # Handle other cases
            abstract = str(abstract_parts)

        # Clean up abstract text
        if abstract:
            # Remove HTML-like tags and clean up
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = re.sub(r'\xa0', ' ', abstract)  # Replace non-breaking spaces
            abstract = re.sub(r'\s+', ' ', abstract).strip()

        journal_info = article.get("Journal", {}).get("Title")
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year") or pub_date.get("MedlineDate")

        authors_list = article.get("AuthorList") or []
        authors: List[str] = []
        for a in authors_list:
            last = a.get("LastName")
            fore = a.get("ForeName") or a.get("Initials")
            if last and fore:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        doi = _extract_doi(entry)
        return PubMedArticle(
            pmid=str(pmid),
            title=title if isinstance(title, str) else str(title),
            abstract=abstract,
            url=url,
            journal=journal_info if isinstance(journal_info, str) else None,
            year=str(year) if year else None,
            authors=authors if authors else None,
            doi=doi,
        )
    except Exception as e:
        print(f"Parsing error: {e}")  # Debug info
        return None


def fetch_pubmed_articles(
    query: str,
    retmax: int = 50,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
    sort: str = "relevance",
) -> List[PubMedArticle]:
    """
    Fetch PubMed articles (title, abstract, url) for a query.
    Uses the exact working pattern from the successful test.
    """
    # Set Entrez parameters
    effective_email = (email or "").strip() or "pubmed-semantic@example.com"
    Entrez.email = effective_email
    Entrez.tool = "pubmed-semantic-app"
    if api_key:
        Entrez.api_key = api_key

    # Step 1: Search PubMed (exactly like the working test)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()

    id_list = record.get("IdList", [])
    if not id_list:
        return []

    # Step 2: Fetch article details (exactly like the working test)
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
    fetch_record = Entrez.read(handle)
    handle.close()

    # Step 3: Parse articles
    articles: List[PubMedArticle] = []
    for entry in fetch_record.get("PubmedArticle", []):
        parsed = _parse_article(entry)
        if parsed:
            articles.append(parsed)

    # Step 4: Enrich with free full-text availability (PMC, then Unpaywall)
    try:
        _enrich_full_text_info(articles, email or "")
    except Exception as e:
        # Do not fail the whole fetch on enrichment errors
        print(f"Enrichment warning: {e}")

    return articles


def _enrich_full_text_info(articles: List[PubMedArticle], email: str) -> None:
    """Populate is_free and full_text_link for each article using PMC and Unpaywall."""
    if not articles:
        return

    pmids = [a.pmid for a in articles if a.pmid]
    pmid_to_article: Dict[str, PubMedArticle] = {a.pmid: a for a in articles}

    # First: try PMC linking in bulk
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(pmids))
        linksets = Entrez.read(handle)
        handle.close()

        # linksets is a list aligned with input IDs, but safer to iterate and map
        for linkset in linksets:
            try:
                id_list = linkset.get("IdList", [])
                src_id = None
                if isinstance(id_list, list) and len(id_list) > 0:
                    src_id = id_list[0].get("Id") if isinstance(id_list[0], dict) else id_list[0]
                if not src_id:
                    continue
                link_dbs = linkset.get("LinkSetDb", [])
                for ldb in link_dbs:
                    # Typical LinkName: "pubmed_pmc"
                    links = ldb.get("Link", [])
                    for link in links:
                        pmcid_num = link.get("Id") if isinstance(link, dict) else None
                        if pmcid_num:
                            pmcid = f"PMC{pmcid_num}"
                            art = pmid_to_article.get(str(src_id))
                            if art and not art.is_free:
                                art.is_free = True
                                art.full_text_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                            break
            except Exception:
                continue
    except Exception:
        # Ignore PMC errors
        pass

    # Second: fallback to Unpaywall for those not free but have DOI
    for art in articles:
        if art.is_free:
            continue
        if not art.doi:
            continue
        try:
            url = f"https://api.unpaywall.org/v2/{art.doi}"
            params = {"email": (email or "pubmed-semantic@example.com")}
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("is_oa"):
                    best = data.get("best_oa_location") or {}
                    link = best.get("url_for_pdf") or best.get("url")
                    if not link:
                        # try any oa_locations
                        for loc in data.get("oa_locations", []) or []:
                            link = loc.get("url_for_pdf") or loc.get("url")
                            if link:
                                break
                    if link:
                        art.is_free = True
                        art.full_text_link = link
        except Exception:
            continue
