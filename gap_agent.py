
import json
import re
from collections import defaultdict
from typing import Any, Literal
from ollama import Client
from pydantic import BaseModel, Field

client = Client(host="http://localhost:11434")

GapCategory = Literal[
    "open_problem",
    "untested_combination",
    "missing_evaluation",
    "contradiction",
]


class GapItem(BaseModel):
    category: GapCategory = Field(
        description="Uma das seguintes categorias: open_problem, untested_combination, missing_evaluation, contradiction"
    )
    title: str
    description: str
    supporting_ids: list[str] = Field(
        description="IDs exatos do OpenAlex ou DOIs dos registos fornecidos que sustentam esta lacuna de investigação"
    )


class ThemeGapReport(BaseModel):
    gaps: list[GapItem] = Field(default_factory=list)


class RankedGapItem(BaseModel):
    rank: int
    title: str
    category: GapCategory
    themes: list[str]
    supporting_ids: list[str]
    rationale: str


class FinalRankedGaps(BaseModel):
    ranked_gaps: list[RankedGapItem] = Field(default_factory=list)


def _normalize_id(val: str) -> str:
    val = (val or "").strip()
    val = re.sub(r"^https?://(dx\.)?doi\.org/", "", val, flags=re.IGNORECASE)
    val = re.sub(r"^https?://openalex\.org/", "", val, flags=re.IGNORECASE)
    return val.strip().lower()


def _extract_paper_ids(paper: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("openalex_id", "id", "doi"):
        raw = paper.get(key)
        if isinstance(raw, str) and raw.strip():
            candidates.append(raw.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        norm = _normalize_id(item)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(item)
    return unique


def _compact_record(paper: dict[str, Any]) -> dict[str, Any]:
    ids = _extract_paper_ids(paper)
    primary_id = ids[0] if ids else "MISSING_ID"

    keywords = paper.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    record: dict[str, Any] = {
        "id_or_doi": ids,
        "primary_id": primary_id,
        "title": (paper.get("title") or "Sem título").strip(),
        "year": paper.get("year") or paper.get("publication_year"),
        "keywords": keywords[:8],
    }

    reason = paper.get("reason")
    if isinstance(reason, str) and reason.strip() and not reason.strip().startswith("Error:"):
        record["summary_reason"] = reason.strip()[:320]

    for optional_key in ("findings", "methods", "limitations", "dataset", "models"):
        if paper.get(optional_key):
            record[optional_key] = str(paper[optional_key])[:280]

    return record


def _group_by_theme(data: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if isinstance(data, dict):
        for theme, papers in data.items():
            if isinstance(papers, list):
                for p in papers:
                    if isinstance(p, dict):
                        grouped[str(theme).strip() or "Geral"].append(p)
        return dict(grouped)

    if isinstance(data, list):
        for paper in data:
            if not isinstance(paper, dict):
                continue
            theme = paper.get("theme") or paper.get("topic") or "Geral"
            if isinstance(theme, list):
                theme = theme[0] if theme else "Geral"
            grouped[str(theme).strip() or "Geral"].append(paper)
        return dict(grouped)

    raise ValueError("O JSON de entrada deve ser uma lista de artigos ou um dicionário que associe temas a listas de artigos.")


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _build_map_prompt(
    theme: str,
    compact_records: list[dict[str, Any]],
    allowed_ids: list[str],
) -> str:
    records_json = json.dumps(compact_records, ensure_ascii=False, indent=2)
    allowed_json = json.dumps(allowed_ids, ensure_ascii=False)

    return f"""
És um agente rigoroso de identificação de lacunas de investigação para o tema: "{theme}".
Analisa APENAS os registos resumidos dos artigos apresentados abaixo (os resumos completos foram intencionalmente omitidos para reduzir o contexto).

Identifica lacunas de investigação nas seguintes quatro categorias:
1. open_problem: Problemas que permanecem por resolver, de forma explícita ou implícita, em vários artigos.
2. untested_combination: Combinações de métodos, abordagens, defesas, tratamentos, culturas ou condições que ainda não foram testadas em conjunto nos artigos fornecidos.
3. missing_evaluation: Avaliações em falta (por exemplo, ausência de benchmarks, dados reais ou de campo, utilização de apenas uma família de modelos, falta de métodos de comparação ou estudos de ablação).
4. contradiction: Contradições diretas ou resultados e conclusões divergentes entre artigos.

REGRA ESTRITA DE FUNDAMENTAÇÃO:
- Cada lacuna DEVE incluir pelo menos um ID ou DOI de suporte presente em ALLOWED_IDS.
- Copia os IDs/DOIs exatamente como aparecem em ALLOWED_IDS.
- Se uma afirmação não tiver um ID de suporte em ALLOWED_IDS, NÃO a incluas. Não inventes artigos nem identificadores.

ALLOWED_IDS:
{allowed_json}

COMPACT_RECORDS:
{records_json}
""".strip()


def _ground_gap_items(
    raw_gaps: list[GapItem],
    valid_id_map: dict[str, str],
) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    for item in raw_gaps:
        matched_ids: list[str] = []
        seen_norm: set[str] = set()
        for cited in item.supporting_ids:
            norm = _normalize_id(cited)
            if norm in valid_id_map and norm not in seen_norm:
                seen_norm.add(norm)
                matched_ids.append(valid_id_map[norm])

        if not matched_ids:
            continue

        grounded.append(
            {
                "category": item.category,
                "title": item.title.strip(),
                "description": item.description.strip(),
                "supporting_ids": matched_ids,
            }
        )
    return grounded


def _build_reduce_prompt(
    theme_gaps: dict[str, list[dict[str, Any]]],
    all_valid_ids: list[str],
) -> str:
    payload_json = json.dumps(theme_gaps, ensure_ascii=False, indent=2)
    allowed_json = json.dumps(all_valid_ids, ensure_ascii=False)

    return f"""
És um investigador sénior responsável pela análise final e consolidação (REDUCE) das lacunas de investigação identificadas em todos os temas.

Abaixo encontram-se as lacunas de investigação fundamentadas para cada tema. Cada lacuna já possui IDs/DOIs de suporte verificados.

Tarefa:
Produz uma única lista ordenada das lacunas de investigação mais importantes e concretas, considerando todos os temas, desde a maior prioridade/impacto até à menor.
- Junta lacunas semelhantes ou praticamente duplicadas entre diferentes temas, sempre que apropriado, combinando os respetivos temas e supporting_ids.
- Cada lacuna classificada DEVE incluir pelo menos um ID de suporte presente em ALLOWED_IDS.
- Fundamenta a justificação exclusivamente nas lacunas identificadas por tema e nos respetivos IDs. Não inventes novos IDs nem afirmações sem suporte.

ALLOWED_IDS:
{allowed_json}

PER_THEME_GROUNDED_GAPS:
{payload_json}
""".strip()


def run_gap_agent(
    input_file: str,
    output_json: str = "research_gaps_ranked.json",
    output_md: str = "research_gaps_report.md",
    model_name: str = "gemma4:12b",
    reduce_model_name: str | None = None,
    num_ctx: int = 8192,
    batch_size: int = 5,
) -> dict[str, Any]:
    reduce_model = reduce_model_name or model_name

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    grouped_papers = _group_by_theme(raw_data)
    all_valid_id_map: dict[str, str] = {}
    per_theme_gaps: dict[str, list[dict[str, Any]]] = {}

    for theme, papers in grouped_papers.items():
        theme_id_map: dict[str, str] = {}
        for p in papers:
            for pid in _extract_paper_ids(p):
                norm = _normalize_id(pid)
                if norm:
                    theme_id_map[norm] = pid
                    all_valid_id_map[norm] = pid

        allowed_ids = list(theme_id_map.values())
        print(f"[MAP] Tema '{theme}': {len(papers)} artigos, {len(allowed_ids)} IDs/DOIs válidos.")

        if not allowed_ids:
            print(f"  -> A ignorar o tema '{theme}': não foram encontrados IDs válidos do OpenAlex ou DOIs.")
            per_theme_gaps[theme] = []
            continue

        compact_records = [_compact_record(p) for p in papers]
        batches = _chunk_list(compact_records, batch_size)
        theme_collected: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for batch_idx, batch_records in enumerate(batches, start=1):
            batch_id_map: dict[str, str] = {}
            for rec in batch_records:
                for raw_id in rec.get("id_or_doi", []):
                    norm = _normalize_id(raw_id)
                    if norm in theme_id_map:
                        batch_id_map[norm] = theme_id_map[norm]

            batch_allowed_ids = list(batch_id_map.values())
            if not batch_allowed_ids:
                continue

            prompt = _build_map_prompt(theme, batch_records, batch_allowed_ids)
            try:
                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    format=ThemeGapReport.model_json_schema(),
                    options={"temperature": 0, "num_ctx": num_ctx},
                )
                parsed = ThemeGapReport.model_validate_json(response.message.content)
                grounded = _ground_gap_items(parsed.gaps, batch_id_map)

                for g in grounded:
                    title_key = g["title"].lower()
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    g["theme"] = theme
                    theme_collected.append(g)

                print(
                    f"  -> Lote {batch_idx}/{len(batches)}: {len(grounded)} lacunas fundamentadas "
                    f"aceites (de {len(parsed.gaps)} identificadas inicialmente)."
                )
            except Exception as e:
                print(f"  -> Erro no tema '{theme}' (lote {batch_idx}/{len(batches)}): {e}")

        per_theme_gaps[theme] = theme_collected

    all_allowed_list = list(all_valid_id_map.values())
    final_ranked: list[dict[str, Any]] = []

    total_grounded = sum(len(v) for v in per_theme_gaps.values())
    if total_grounded > 0:
        print(
            f"\n[REDUCE] A ordenar {total_grounded} lacunas fundamentadas em {len(per_theme_gaps)} temas "
            f"utilizando o modelo '{reduce_model}'..."
        )
        reduce_prompt = _build_reduce_prompt(per_theme_gaps, all_allowed_list)
        try:
            reduce_resp = client.chat(
                model=reduce_model,
                messages=[{"role": "user", "content": reduce_prompt}],
                format=FinalRankedGaps.model_json_schema(),
                options={"temperature": 0, "num_ctx": num_ctx},
            )
            parsed_reduce = FinalRankedGaps.model_validate_json(reduce_resp.message.content)

            for idx, item in enumerate(parsed_reduce.ranked_gaps, start=1):
                verified_ids: list[str] = []
                seen: set[str] = set()
                for cid in item.supporting_ids:
                    norm = _normalize_id(cid)
                    if norm in all_valid_id_map and norm not in seen:
                        seen.add(norm)
                        verified_ids.append(all_valid_id_map[norm])

                if verified_ids:
                    final_ranked.append(
                        {
                            "rank": idx,
                            "title": item.title.strip(),
                            "category": item.category,
                            "themes": [t.strip() for t in item.themes if t.strip()],
                            "supporting_ids": verified_ids,
                            "rationale": item.rationale.strip(),
                        }
                    )
        except Exception as e:
            print(f"[REDUCE] Erro: {e}")

    result_payload = {
        "map_model": model_name,
        "reduce_model": reduce_model,
        "total_themes": len(grouped_papers),
        "total_grounded_theme_gaps": total_grounded,
        "ranked_gaps": final_ranked,
        "per_theme_gaps": per_theme_gaps,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Relatório de Lacunas de Investigação Classificadas",
        "",
        f"- **Modelo de identificação:** `{model_name}`",
        f"- **Modelo de consolidação:** `{reduce_model}`",
        f"- **Temas analisados:** {len(grouped_papers)}",
        f"- **Lacunas fundamentadas antes da consolidação:** {total_grounded}",
        f"- **Lacunas finais classificadas:** {len(final_ranked)}",
        "",
        "## Lacunas de Investigação Classificadas",
        "",
    ]

    for gap in final_ranked:
        themes_str = ", ".join(gap["themes"]) if gap["themes"] else "Geral"
        refs_str = ", ".join(f"`{rid}`" for rid in gap["supporting_ids"])
        md_lines.extend(
            [
                f"### {gap['rank']}. {gap['title']}",
                f"- **Categoria:** `{gap['category']}`",
                f"- **Temas:** {themes_str}",
                f"- **Artigos de suporte (ID / DOI):** {refs_str}",
                f"- **Justificação:** {gap['rationale']}",
                "",
            ]
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nFicheiros guardados: {output_json} e {output_md}")
    return result_payload


if __name__ == "__main__":
    run_gap_agent(
        input_file="evaluated_dataset.json",
        output_json="research_gaps_ranked.json",
        output_md="research_gaps_report.md",
        model_name="gemma4:12b",
        reduce_model_name=None,
        num_ctx=8192,
        batch_size=5,
    )
