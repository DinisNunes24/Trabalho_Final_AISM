import json
import os
import pyalex
from pyalex import Works

# Setting an email puts you in the "polite pool" for faster, more reliable rate limits
pyalex.config.email = "240000642@esg.ipsantarem.pt"

pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

if not pyalex.config.api_key:
    raise ValueError("Must set the OPENALEX_API_KEY environment variable")


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    OpenAlex returns abstracts as an inverted index to save space.
    Format: {"The": [0, 5], "moth": [1], ...}
    This function rebuilds the original string.
    """
    if not inverted_index:
        return ""

    # Collect valid non-negative positions safely (avoids ValueError if any list is empty)
    positions_all = [
        pos
        for positions in inverted_index.values()
        if isinstance(positions, list)
        for pos in positions
        if isinstance(pos, int) and pos >= 0
    ]
    if not positions_all:
        return ""

    # Find the length of the abstract
    max_index = max(positions_all)

    # Initialize an array of empty strings
    words = [""] * (max_index + 1)

    # Place each word at its specified indices
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos <= max_index:
                words[pos] = word

    # Filter empty slots to avoid double spaces if any index is missing
    return " ".join(w for w in words if w)


def fetch_papers(query: str, output_file: str = "papers_dataset.json", per_page: int = 20):
    print(f"Searching OpenAlex for: '{query}'...")

    # We use .search() to query the full text, title, and abstract
    # .select() limits the payload size
    pager = (
        Works()
        .search(query)
        .select([
            "id",
            "title",
            "publication_year",
            "authorships",
            "abstract_inverted_index",
            "doi",
            "keywords",
        ])
        .paginate(per_page=per_page)
    )

    dataset = []

    # Iterate through pages (save partial progress if interrupted/errored)
    try:
        for page_num, page in enumerate(pager, start=1):
            print(f"Fetching page {page_num}... ({len(page)} records)")

            for work in page:
                # Parse authors into a clean list safely
                authors = [
                    author["author"].get("display_name")
                    for author in (work.get("authorships") or [])
                    if isinstance(author, dict) and isinstance(author.get("author"), dict)
                ]
                authors = [name for name in authors if name]

                # Reconstruct the abstract (returns "" if None)
                abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))
                title_text = work.get("title") or "No Title Available"

                keywords = []
                for keyword_entry in work.get("keywords") or []:
                    if isinstance(keyword_entry, dict):
                        keyword_name = keyword_entry.get("display_name")
                        if keyword_name:
                            keywords.append(keyword_name)

                has_abstract = bool(abstract_text.strip())

                dataset.append({
                    "id": work.get("id"),
                    "doi": work.get("doi"),
                    "title": title_text,
                    "year": work.get("publication_year"),
                    "authors": authors,
                    "keywords": keywords,
                    "has_abstract": has_abstract,
                    "abstract": abstract_text,
                })
    finally:
        # Save to disk even if pagination stops early due to an error
        if dataset:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"\nDone! Saved {len(dataset)} records to {output_file}.")
        else:
            print("\nDone! No records to save.")


if __name__ == "__main__":
    # Run the extraction
    fetch_papers(query="Cryptoblabes gnidiella", output_file="papers_dataset.json")