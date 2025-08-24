from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass
from Bio import Entrez
import requests
import re


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
    free_source: Optional[str] = None   # "pmc" | "publisher" | "unpaywall"


def _extract_doi(entry: dict) -> Optional[str]:
    """Extract DOI if available."""
    try:
        medline = entry.get("MedlineCitation", {})
        article = medline.get("Article", {})

        # Option 1: ELocationID with EIdType = doi
        eids = article.get("ELocationID")
        if isinstance(eids, list):
            for eid in eids:
                if isinstance(eid, dict) and eid.get("@EIdType", "").lower() == "doi":
                    return str(eid.get("#text")).strip()
        elif isinstance(eids, dict):
            if eids.get("@EIdType", "").lower() == "doi":
                return str(eids.get("#text")).strip()

        # Option 2: PubmedData -> ArticleIdList
        pubmed_data = entry.get("PubmedData", {})
        id_list = pubmed_data.get("ArticleIdList") or []
        if isinstance(id_list, list):
            for aid in id_list:
                if isinstance(aid, dict) and aid.get("@IdType", "").lower() == "doi":
                    return str(aid.get("#text")).strip()
        elif isinstance(id_list, dict):
            if id_list.get("@IdType", "").lower() == "doi":
                return str(id_list.get("#text")).strip()
    except Exception:
        return None
    return None


def _parse_article(entry: dict) -> Optional[PubMedArticle]:
    """Convert XML entry to PubMedArticle object."""
    try:
        medline = entry.get("MedlineCitation", {})
        pmid_raw = medline.get("PMID")
        pmid = pmid_raw.get("#text") if isinstance(pmid_raw, dict) else pmid_raw
        if not pmid:
            return None

        article = medline.get("Article", {})
        title = article.get("ArticleTitle") or ""

        # Handle abstract
        abstract_parts = article.get("Abstract", {}).get("AbstractText")
        abstract = ""
        if isinstance(abstract_parts, list):
            abstract = " ".join(str(part.content if hasattr(part, "content") else part) for part in abstract_parts)
        elif isinstance(abstract_parts, str):
            abstract = abstract_parts
        elif abstract_parts:
            abstract = str(abstract_parts)

        # Clean abstract
        abstract = re.sub(r"<[^>]+>", "", abstract or "")
        abstract = re.sub(r"\xa0", " ", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()

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
        print(f"Parsing error: {e}")
        return None


def _enrich_full_text_info(articles: List[PubMedArticle], email: str) -> None:
    """Mark articles as free if available in PMC or via Unpaywall."""
    if not articles:
        return

    pmids = [a.pmid for a in articles if a.pmid]
    pmid_to_article: Dict[str, PubMedArticle] = {a.pmid: a for a in articles}

    # First: check PMC linking
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(pmids))
        linksets = Entrez.read(handle)
        handle.close()
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
                    links = ldb.get("Link", [])
                    for link in links:
                        pmcid_num = link.get("Id") if isinstance(link, dict) else None
                        if pmcid_num:
                            pmcid = f"PMC{pmcid_num}"
                            art = pmid_to_article.get(str(src_id))
                            if art and not art.is_free:
                                art.is_free = True
                                art.full_text_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                                art.free_source = "pmc"
                            break
            except Exception:
                continue
    except Exception:
        pass

    # Second: check DOI via Unpaywall
    for art in articles:
        if art.is_free:  # already covered by PMC
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
                        for loc in data.get("oa_locations", []) or []:
                            link = loc.get("url_for_pdf") or loc.get("url")
                            if link:
                                break
                    if link:
                        art.is_free = True
                        art.full_text_link = link
                        art.free_source = best.get("host_type", "unpaywall")
        except Exception:
            continue


def fetch_pubmed_articles(
    query: str,
    retmax: int = 50,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
    sort: str = "relevance",
    free_only: bool = False,   # <-- NEW FLAG
) -> List[PubMedArticle]:
    """
    Fetch PubMed articles (title, abstract, url).
    If free_only=True → returns only free articles (PMC, publisher, or Unpaywall).
    If free_only=False → returns all articles, but marks free ones.
    """
    effective_email = (email or "").strip() or "pubmed-semantic@example.com"
    Entrez.email = effective_email
    Entrez.tool = "pubmed-semantic-app"
    if api_key:
        Entrez.api_key = api_key

    # Step 1: Search PubMed (do NOT apply free filter here)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort)
    record = Entrez.read(handle)
    handle.close()

    id_list = record.get("IdList", [])
    if not id_list:
        return []

    # Step 2: Fetch article details
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
    fetch_record = Entrez.read(handle)
    handle.close()

    # Step 3: Parse articles
    articles: List[PubMedArticle] = []
    for entry in fetch_record.get("PubmedArticle", []):
        parsed = _parse_article(entry)
        if parsed:
            articles.append(parsed)

    # Step 4: Enrich with free info (PMC + Unpaywall)
    try:
        _enrich_full_text_info(articles, email or "")
    except Exception as e:
        print(f"Enrichment warning: {e}")

    # Step 5: Filter if user wants only free ones
    if free_only:
        articles = [a for a in articles if a.is_free]

    return articles


# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    results = fetch_pubmed_articles("breast cancer", retmax=5, email="your_email@example.com")
    for art in results:
        print(f"\nTitle: {art.title}")
        print(f"PMID: {art.pmid}")
        print(f"DOI: {art.doi}")
        print(f"Free Source: {art.free_source}")
        print(f"Link: {art.full_text_link}")
