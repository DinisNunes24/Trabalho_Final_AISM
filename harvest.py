import json
import os
from collections import Counter

import pyalex
from pyalex import Works, Topics, Keywords
from dotenv import load_dotenv

# Configuração
pyalex.config.email = "240000642@esg.ipsantarem.pt"
load_dotenv()
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

if not pyalex.config.api_key:
    raise ValueError("Must set the OPENALEX_API_KEY environment variable")


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstrói o abstract a partir do inverted index do OpenAlex."""
    if not inverted_index:
        return ""

    positions_all = [
        pos
        for positions in inverted_index.values()
        if isinstance(positions, list)
        for pos in positions
        if isinstance(pos, int) and pos >= 0
    ]
    if not positions_all:
        return ""

    max_index = max(positions_all)
    words = [""] * (max_index + 1)

    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos <= max_index:
                words[pos] = word

    return " ".join(w for w in words if w)


def suggest_related_topics(query: str, max_suggestions: int = 8):
    """
    Sugere temas relacionados usando Topics e Keywords do OpenAlex.
    """
    print(f"\n A procurar temas relacionados com: '{query}'...")
    suggestions = []

    try:
        topics = Topics().search(query).get(per_page=5)
        for t in topics:
            name = t.get("display_name")
            if name:
                suggestions.append(("Topic", name))
    except Exception as e:
        print(f"   (Aviso: não foi possível obter Topics: {e})")

    try:
        keywords = Keywords().search(query).get(per_page=5)
        for k in keywords:
            name = k.get("display_name")
            if name:
                suggestions.append(("Keyword", name))
    except Exception as e:
        print(f"   (Aviso: não foi possível obter Keywords: {e})")

    seen = set()
    unique = []
    for tipo, nome in suggestions:
        chave = nome.lower()
        if chave not in seen:
            seen.add(chave)
            unique.append((tipo, nome))

    if unique:
        print("\n Temas relacionados sugeridos:")
        for i, (tipo, nome) in enumerate(unique[:max_suggestions], 1):
            print(f"   {i}. [{tipo}] {nome}")
    else:
        print("   Não foram encontrados temas relacionados.")

    return [nome for _, nome in unique[:max_suggestions]]


def choose_topic() -> str | None:
    """Pede o tema ao utilizador, sugere relacionados e devolve o tema final."""
    query = input("\nDigite o tema que pretende pesquisar: ").strip()

    if not query:
        print("Nenhum tema introduzido.")
        return None

    related = suggest_related_topics(query)

    if related:
        print("\nQuer usar um dos temas sugeridos em vez do original?")
        print("   (pressione Enter para manter o tema original)")
        escolha = input("   Número do tema (ou Enter): ").strip()

        if escolha.isdigit():
            idx = int(escolha) - 1
            if 0 <= idx < len(related):
                query = related[idx]
                print(f"\n→ Tema alterado para: '{query}'")

    return query


def build_output_filename(query: str, prefix: str = "papers") -> str:
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in query)[:50]
    clean_name = safe_name.strip().replace(" ", "_")
    return f"{prefix}_{clean_name}.json"


def fetch_papers(query: str, output_file: str = "papers_dataset.json", per_page: int = 20, max_results: int = 20):
    print(f"\n A pesquisar no OpenAlex: '{query}'...")

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
            "primary_topic",
        ])
        .paginate(per_page=per_page)
    )

    dataset = []
    all_keywords = []

    try:
        for page_num, page in enumerate(pager, start=1):
            print(f"   Página {page_num}... ({len(page)} registos)")

            for work in page:
                authors = [
                    author["author"].get("display_name")
                    for author in (work.get("authorships") or [])
                    if isinstance(author, dict) and isinstance(author.get("author"), dict)
                ]
                authors = [name for name in authors if name]

                abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))
                title_text = work.get("title") or "No Title Available"

                keywords = []
                for keyword_entry in work.get("keywords") or []:
                    if isinstance(keyword_entry, dict):
                        keyword_name = keyword_entry.get("display_name")
                        if keyword_name:
                            keywords.append(keyword_name)
                            all_keywords.append(keyword_name)

                primary_topic = None
                if work.get("primary_topic"):
                    primary_topic = work["primary_topic"].get("display_name")

                dataset.append({
                    "id": work.get("id"),
                    "doi": work.get("doi"),
                    "title": title_text,
                    "year": work.get("publication_year"),
                    "authors": authors,
                    "keywords": keywords,
                    "primary_topic": primary_topic,
                    "has_abstract": bool(abstract_text.strip()),
                    "abstract": abstract_text,
                })

                if len(dataset) >= max_results:
                    break

            if len(dataset) >= max_results:
                break

    finally:
        if dataset:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"\n Guardados {len(dataset)} registos em '{output_file}'.")

            if all_keywords:
                print("\n Keywords mais frequentes nos resultados:")
                for kw, count in Counter(all_keywords).most_common(8):
                    print(f"   • {kw} ({count}x)")
        else:
            print("\n Nenhum registo encontrado.")


if __name__ == "__main__":
    print("=" * 60)
    print("  Pesquisa de artigos científicos no OpenAlex")
    print("=" * 60)

    selected_query = choose_topic()
    if not selected_query:
        print("A terminar.")
        exit()

    output_file = build_output_filename(selected_query, prefix="papers")
    fetch_papers(query=selected_query, output_file=output_file, per_page=20, max_results=20)