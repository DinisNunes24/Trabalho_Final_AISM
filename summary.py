import json
import os
from ollama import Client

from harvest import build_output_filename, choose_topic, fetch_papers

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
    input_file: str | None = None,
    output_file: str | None = None,
    model_name: str = "gemma3:4b",
    save_every: int = 25,
    fetch_if_missing: bool = True,
):
    if not topic:
        topic = choose_topic()
    if not topic:
        print("Nenhum tema selecionado. A terminar.")
        return

    if not input_file:
        input_file = build_output_filename(topic, prefix="papers")

    if not output_file:
        output_file = build_output_filename(topic, prefix="evaluated")

    if not os.path.exists(input_file):
        if fetch_if_missing:
            print(f"Ficheiro '{input_file}' não encontrado. A recolher artigos para '{topic}'...")
            fetch_papers(query=topic, output_file=input_file, per_page=20, max_results=20)
        else:
            raise FileNotFoundError(f"Ficheiro de entrada não encontrado: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} papers from '{input_file}'. Evaluating against topic: '{topic}' using {model_name}...")

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

        if (index + 1) % save_every == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nFinished! Evaluated dataset saved to {output_file}")


if __name__ == "__main__":
    evaluate_papers()