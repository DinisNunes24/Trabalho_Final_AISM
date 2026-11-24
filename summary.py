import json
import os
from ollama import Client

from harvest import choose_topic

client = Client(host="http://localhost:11434")


def _build_prompt(paper: dict, topic: str) -> str:
    title = paper.get("title") or "Unavailable"
    keywords = paper.get("keywords") or []
    keywords_text = ", ".join(keywords) if keywords else "Unavailable"
    primary_topic = paper.get("primary_topic") or "Unavailable"
    abstract = paper.get("abstract") or "Unavailable"

    return f"""
You are a rigorous academic research analyst. Analyze the following scientific paper regarding its connection to the topic "{topic}".

Explain clearly and concisely whether and how the paper relates to "{topic}" (for example: core focus, methods, applications, context, or lack of direct relevance), referencing evidence from the title, keywords, primary topic, or abstract.

TOPIC UNDER EVALUATION:
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
    topic: str | None = None,
    input_file: str = "papers_dataset.json",
    output_file: str = "evaluated_dataset.json",
    model_name: str = "gemma3:4b",
    save_every: int = 25,
):
    # Tentar obter o tema automaticamente do ficheiro criado pelo harvest.py
    if not topic:
        if os.path.exists("last_topic.txt"):
            with open("last_topic.txt", "r", encoding="utf-8") as f:
                topic = f.read().strip()
            print(f"Tema detetado automaticamente: '{topic}'")
        else:
            topic = choose_topic()  # só pergunta se não existir o ficheiro

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
            print(f"Error evaluating paper {paper.get('id')}: {e}")
            paper["reason"] = f"Error: {e}"

        # Guardar periodicamente
        if (index + 1) % save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"   → Checkpoint guardado em {output_file}")

    # Guardar resultado final
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nFinished! Evaluated dataset saved to {output_file}")


if __name__ == "__main__":
    evaluate_papers()