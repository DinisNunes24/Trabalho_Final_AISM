import json
import os
from ollama import Client

from harvest import choose_topic_stack

client = Client(host="http://localhost:11434")


def _normalize_topic(topic) -> str | None:
    """Garante que o tema é sempre uma string legível."""
    if topic is None:
        return None
    if isinstance(topic, list):
        # junta com " | " para manter legibilidade no prompt
        cleaned = [t.strip() for t in topic if t and t.strip()]
        return " | ".join(cleaned) if cleaned else None
    if isinstance(topic, str):
        return topic.strip() or None
    return str(topic)


def _build_prompt(paper: dict, topic: str) -> str:
    title = paper.get("title") or "Unavailable"
    keywords = paper.get("keywords") or []
    keywords_text = ", ".join(keywords) if keywords else "Unavailable"
    primary_topic = paper.get("primary_topic") or "Unavailable"
    abstract = paper.get("abstract") or "Unavailable"

    return f"""

És um analista rigoroso de investigação académica. Analisa o seguinte artigo científico relativamente à sua relação com o tema "{topic}".

Explica de forma clara e concisa se, e de que forma, o artigo está relacionado com o tema "{topic}" (por exemplo: foco principal, métodos, aplicações, contexto ou ausência de relevância direta), fundamentando a tua análise com evidências retiradas do título, das palavras-chave, do tema principal ou do resumo (abstract).

TEMA EM AVALIAÇÃO
{topic}

TITLE:
{title}

PRIMARY TOPIC:
{primary_topic}

KEYWORDS:
{keywords_text}

ABSTRACT:
{abstract}
""".strip()


def evaluate_papers(
    topic: str | list[str] | None = None,
    input_file: str = "papers_dataset.json",
    output_file: str = "evaluated_dataset.json",
    model_name: str = "gemma3:4b",
    save_every: int = 25,
):
    # 1. Tentar obter o tema automaticamente
    if not topic:
        # Preferir o ficheiro JSON da stack (mais fiável)
        if os.path.exists("last_topics.json"):
            with open("last_topics.json", "r", encoding="utf-8") as f:
                topic = json.load(f)
            print(f"Temas detetados automaticamente (stack): {topic}")
        elif os.path.exists("last_topic.txt"):
            with open("last_topic.txt", "r", encoding="utf-8") as f:
                topic = f.read().strip()
            print(f"Tema detetado automaticamente: '{topic}'")
        else:
            # Só pergunta se não existir nenhum ficheiro
            topic = choose_topic_stack()

    # 2. Normalizar para string
    topic = _normalize_topic(topic)

    if not topic:
        print("Nenhum tema selecionado. A terminar.")
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Ficheiro de entrada não encontrado: {input_file}\n"
            "Execute primeiro o harvest.py para gerar o ficheiro."
        )

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} papers from '{input_file}'.")
    print(f"Evaluating against topic: '{topic}' using {model_name}...")

    for index, paper in enumerate(dataset):
        title_preview = (paper.get("title") or "Untitled")[:50]
        print(f"[{index + 1}/{len(dataset)}] Evaluating: {title_preview}...")

        prompt = _build_prompt(paper, topic=topic)
        paper["evaluated_topic"] = topic
        paper["prompt_text"] = prompt

        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                },
            )
            paper["reason"] = response.message.content.strip()

        except Exception as e:
            print(f"Erro ao avaliar {paper.get('id')}: {e}")
            paper["reason"] = f"Error: {e}"

        # Guardar periodicamente
        if (index + 1) % save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"   → Checkpoint guardado em {output_file}")

    # Guardar resultado final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nAcabado! Evaluated dataset salavdo para {output_file}")


if __name__ == "__main__":
    evaluate_papers()