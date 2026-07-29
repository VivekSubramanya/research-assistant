import logging
import os
import feedparser
from datetime import datetime
import time

import requests

BASE_URL = "http://export.arxiv.org/api/query"
PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_CONNECT_TIMEOUT_SECONDS = 10
ARXIV_READ_TIMEOUT_SECONDS = 30
ARXIV_RETRY_ATTEMPTS = 3
ARXIV_RETRY_BACKOFF_SECONDS = 2.0


class ArxivQueryError(RuntimeError):
    def __init__(self, message, *, is_temporary=False):
        super().__init__(message)
        self.is_temporary = is_temporary


def download_arxiv_pdf(arxiv_id, file_path):
    """Download a paper PDF from arXiv and save it to file_path."""
    logger = logging.getLogger("job")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    url = PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    logger.info(f"Downloading arXiv PDF from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Saved arXiv PDF to {file_path}")
    return file_path


def query_arxiv(params: dict):
    """
    Generic arXiv query executor.
    Params is a dict of query parameters (search_query, start, max_results, etc.)
    """
    logger = logging.getLogger("job")
    last_error = None
    for attempt in range(1, ARXIV_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                timeout=(ARXIV_CONNECT_TIMEOUT_SECONDS, ARXIV_READ_TIMEOUT_SECONDS),
            )
            logger.info(f"arXiv API request URL={response.url}")
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            logger.info(f"arXiv API response entries={len(feed.entries)}")
            return feed.entries
        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "stage=arxiv_api_retry attempt=%s/%s reason=%s",
                attempt,
                ARXIV_RETRY_ATTEMPTS,
                "timeout",
            )
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code and status_code >= 500 and attempt < ARXIV_RETRY_ATTEMPTS:
                logger.warning(
                    "stage=arxiv_api_retry attempt=%s/%s reason=%s status_code=%s",
                    attempt,
                    ARXIV_RETRY_ATTEMPTS,
                    "server_error",
                    status_code,
                )
            else:
                break
        except requests.RequestException as exc:
            last_error = exc
            break

        if attempt < ARXIV_RETRY_ATTEMPTS:
            time.sleep(ARXIV_RETRY_BACKOFF_SECONDS * attempt)

    if isinstance(last_error, requests.exceptions.Timeout):
        raise ArxivQueryError(
            "arXiv is taking too long to respond right now. Please try again shortly.",
            is_temporary=True,
        ) from last_error

    if isinstance(last_error, requests.exceptions.HTTPError):
        status_code = last_error.response.status_code if last_error.response is not None else None
        if status_code and status_code >= 500:
            raise ArxivQueryError(
                "arXiv is temporarily unavailable right now. Please try again shortly.",
                is_temporary=True,
            ) from last_error
        raise ArxivQueryError(
            "The arXiv request could not be completed for this query.",
            is_temporary=False,
        ) from last_error

    if last_error is not None:
        raise ArxivQueryError(
            "The arXiv request failed before results could be retrieved.",
            is_temporary=False,
        ) from last_error

    return []


def parse_entry(entry):
    """
    Extract metadata from a single arXiv entry.
    """
    published_raw = entry.get("published")
    published_dt = None
    if published_raw:
        try:
            published_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except Exception:
            published_dt = published_raw

    authors = entry.get("authors", [])
    if authors and hasattr(authors[0], "name"):
        author_names = [author.name for author in authors]
    else:
        author_names = [author.get("name") if isinstance(author, dict) else str(author) for author in authors]

    tags = entry.get("tags", [])
    categories = [tag["term"] if isinstance(tag, dict) else str(tag) for tag in tags]

    links = entry.get("links", [])
    pdf_url = None
    if links:
        pdf_link = next((link.get("href") for link in links if isinstance(link, dict) and link.get("title") == "pdf"), None)
        if pdf_link is None:
            pdf_link = next((link.href for link in links if getattr(link, "title", None) == "pdf"), None)
        pdf_url = pdf_link

    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "summary": entry.get("summary"),
        "authors": author_names,
        "published": published_dt,
        "categories": categories,
        "pdf_url": pdf_url,
    }


class PaperRecord:
    def __init__(self, entry_data):
        self._data = entry_data

    @property
    def title(self):
        return self._data.get("title")

    @property
    def summary(self):
        return self._data.get("summary")

    @property
    def authors(self):
        return self._data.get("authors", [])

    @property
    def categories(self):
        return self._data.get("categories", [])

    @property
    def published(self):
        return self._data.get("published")

    @property
    def pdf_url(self):
        return self._data.get("pdf_url")


class Search:
    def __init__(self, id_list=None, query=None, max_results=10):
        self.id_list = id_list or []
        self.query = query or ""
        self.max_results = max_results

    def results(self):
        params = {"start": 0, "max_results": self.max_results}
        if self.id_list:
            params["id_list"] = ",".join(self.id_list)
        if self.query:
            params["search_query"] = self.query

        entries = query_arxiv(params)
        for entry in entries[: self.max_results]:
            yield PaperRecord(parse_entry(entry))
