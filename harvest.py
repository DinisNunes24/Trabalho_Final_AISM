import json
import os
from collections import Counter

import pyalex
from pyalex import Works, Topics, Keywords
from dotenv import load_dotenv


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


def suggest_related_topics(query: str, max_suggestions: int = 8) -> list[str]:
    """
    Sugere temas relacionados usando Topics e Keywords do OpenAlex.
    Devolve apenas a lista de nomes (sem imprimir).
    """
    suggestions = []

    # 1. Topics
    try:
        topics = Topics().search(query).get(per_page=5)
        for t in topics:
            name = t.get("display_name")
            if name:
                suggestions.append(name)
    except Exception as e:
        print(f"   (Aviso: não foi possível obter Topics: {e})")

    # 2. Keywords
    try:
        keywords = Keywords().search(query).get(per_page=5)
        for k in keywords:
            name = k.get("display_name")
            if name:
                suggestions.append(name)
    except Exception as e:
        print(f"   (Aviso: não foi possível obter Keywords: {e})")

    # Remover duplicados mantendo a ordem
    seen = set()
    unique = []
    for nome in suggestions:
        chave = nome.lower()
        if chave not in seen:
            seen.add(chave)
            unique.append(nome)

    return unique[:max_suggestions]


def choose_topic_stack() -> list[str] | None:
    """
    Permite ao utilizador construir, editar e confirmar uma stack de temas.
    Só devolve a lista quando o utilizador escolhe 'search'.
    """
    topics: list[str] = []

    print("=" * 60)
    print("  Construtor de stack de temas")
    print("=" * 60)
    print("""
Comandos disponíveis:
  add <texto>          -> adiciona um tema livre
  suggest <texto>      -> obtém sugestões do OpenAlex e escolhe uma
  remove <n>           -> remove o tema número n
  clear                -> limpa toda a stack
  show                 -> mostra a stack actual
  search               -> confirma e pesquisa no OpenAlex
  quit / q             -> sair sem pesquisar
""")

    while True:
        print("\nStack actual:", topics if topics else "(vazia)")
        raw = input("> ").strip()
        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        #add 
        if cmd == "add":
            if not arg:
                print("Uso: add <texto do tema>")
                continue
            if arg.lower() in (t.lower() for t in topics):
                print(f"  '{arg}' já está na stack.")
            else:
                topics.append(arg)
                print(f"  + Adicionado: '{arg}'")

        # suggest 
        elif cmd == "suggest":
            if not arg:
                print("Uso: suggest <texto>")
                continue

            print(f"\n A procurar sugestões para: '{arg}'...")
            suggestions = suggest_related_topics(arg)

            if not suggestions:
                print("   Nenhuma sugestão encontrada.")
                continue

            print("\n Sugestões:")
            for i, nome in enumerate(suggestions, 1):
                print(f"   {i}. {nome}")

            escolha = input("\nNúmero da sugestão a adicionar (ou Enter para cancelar): ").strip()
            if escolha.isdigit():
                idx = int(escolha) - 1
                if 0 <= idx < len(suggestions):
                    escolhido = suggestions[idx]
                    if escolhido.lower() in (t.lower() for t in topics):
                        print(f"  '{escolhido}' já está na stack.")
                    else:
                        topics.append(escolhido)
                        print(f"  + Adicionado: '{escolhido}'")
                else:
                    print("  Número inválido.")
            else:
                print("  Sugestão cancelada.")

        # remove
        elif cmd == "remove":
            if not arg:
                print("Uso: remove <número>")
                continue
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(topics):
                    removed = topics.pop(idx)
                    print(f"  - Removido: '{removed}'")
                else:
                    print("  Número fora do intervalo.")
            except ValueError:
                print("  Número inválido.")

        # clear
        elif cmd == "clear":
            topics.clear()
            print("  Stack limpa.")

        # show 
        elif cmd == "show":
            if not topics:
                print("  (vazia)")
            else:
                for i, t in enumerate(topics, 1):
                    print(f"  {i}. {t}")

        #search (confirmação) 
        elif cmd == "search":
            if not topics:
                print("  A stack está vazia. Adicione pelo menos um tema.")
                continue
            print(f"\n→ A pesquisar com a stack: {topics}")
            return topics

        # quit
        elif cmd in ("quit", "q", "exit"):
            print("  A sair sem pesquisar.")
            return None

        else:
            print("  Comando desconhecido. Tente: add, suggest, remove, clear, show, search, quit")


def fetch_papers(query: str,
                 output_file: str = "papers_dataset.json",
                 per_page: int = 20,
                 max_results: int = 20):
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
            print(f"  Página {page_num}: {len(page)} registos")

            for work in page:
                # Autores
                authors = [
                    author["author"].get("display_name")
                    for author in (work.get("authorships") or [])
                    if isinstance(author, dict) and isinstance(author.get("author"), dict)
                ]
                authors = [name for name in authors if name]

                # Abstract
                abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))
                title_text = work.get("title") or "No Title Available"

                # Keywords do artigo
                keywords = []
                for keyword_entry in work.get("keywords") or []:
                    if isinstance(keyword_entry, dict):
                        keyword_name = keyword_entry.get("display_name")
                        if keyword_name:
                            keywords.append(keyword_name)
                            all_keywords.append(keyword_name)

                # Topic principal
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
    topics = choose_topic_stack()

    if not topics:
        print("Nenhum tema confirmado. A terminar.")
    else:
        # Guardar a stack completa (útil para scripts posteriores)
        with open("last_topics.json", "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)

        # Também guardar uma string simples (compatibilidade com código antigo)
        with open("last_topic.txt", "w", encoding="utf-8") as f:
            f.write(" | ".join(topics))

        # Combinar os temas com AND (pode mudar para OR se preferir)
        combined_query = " AND ".join(f'"{t}"' for t in topics)

        fetch_papers(
            query=combined_query,
            output_file="papers_dataset.json",
            per_page=20,
            max_results=20
        )