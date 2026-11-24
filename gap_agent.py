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
        description="One of: open_problem, untested_combination, missing_evaluation, contradiction"
    )
    title: str
    description: str
    supporting_ids: list[str] = Field(
        description="Exact OpenAlex IDs or DOIs from the provided records that support this gap"
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
        "title": (paper.get("title") or "Untitled").strip(),
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
                        grouped[str(theme).strip() or "General"].append(p)
        return dict(grouped)

    if isinstance(data, list):
        for paper in data:
            if not isinstance(paper, dict):
                continue
            theme = paper.get("theme") or paper.get("topic") or "General"
            if isinstance(theme, list):
                theme = theme[0] if theme else "General"
            grouped[str(theme).strip() or "General"].append(paper)
        return dict(grouped)

    raise ValueError("Input JSON must be a list of paper dicts or a dict mapping theme -> list of papers.")


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
You are a rigorous research gap analysis agent for the theme: "{theme}".
Analyze ONLY the compact paper records below (full abstracts are intentionally omitted to save context).

Identify research gaps across these four categories:
1. open_problem: Problems explicitly or implicitly left open across multiple papers.
2. untested_combination: Combinations of methods, attacks, defenses, treatments, crops, or conditions nobody in these records has tried together.
3. missing_evaluation: Missing evaluations (e.g., no benchmark, no real-world/field data, only one model family, lack of baselines or ablation).
4. contradiction: Direct contradictions or conflicting findings/claims between papers.

STRICT GROUNDING RULE:
- Every gap MUST cite at least one supporting ID or DOI from ALLOWED_IDS.
- Copy supporting IDs/DOIs verbatim from ALLOWED_IDS.
- If a claim has no supporting ID in ALLOWED_IDS, DO NOT include it. Do not invent papers or IDs.

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
You are a senior research strategist performing a final REDUCE across all themes.
Below are grounded research gaps per theme. Each gap already has verified supporting IDs/DOIs.

Task:
Produce a single ranked list of the most important, concrete research gaps across all themes (ordered from highest priority/impact to lowest).
- Merge near-duplicate gaps across themes when appropriate, combining their themes and supporting_ids.
- Every ranked gap MUST include at least one supporting ID from ALLOWED_IDS.
- Base your rationale strictly on the provided theme gaps and cited IDs; do not invent new IDs or unsupported claims.

ALLOWED_IDS:
{allowed_json}

PER_THEME_GROUNDED_GAPS:
{payload_json}
""".strip()


def run_gap_agent(
    input_file: str,
    output_json: str = "research_gaps_ranked.json",
    output_md: str = "research_gaps_report.md",
    model_name: str = "gemma3:4b",
    reduce_model_name: str | None = None,
    num_ctx: int = 4096,
    batch_size: int = 20,
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
        print(f"[MAP] Theme '{theme}': {len(papers)} papers, {len(allowed_ids)} valid IDs/DOIs.")

        if not allowed_ids:
            print(f"  -> Skipping theme '{theme}': no valid OpenAlex ID or DOI found.")
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
                    f"  -> Batch {batch_idx}/{len(batches)}: kept {len(grounded)} grounded gaps "
                    f"(from {len(parsed.gaps)} raw)."
                )
            except Exception as e:
                print(f"  -> Error in theme '{theme}' (batch {batch_idx}/{len(batches)}): {e}")

        per_theme_gaps[theme] = theme_collected

    all_allowed_list = list(all_valid_id_map.values())
    final_ranked: list[dict[str, Any]] = []

    total_grounded = sum(len(v) for v in per_theme_gaps.values())
    if total_grounded > 0:
        print(
            f"\n[REDUCE] Ranking {total_grounded} grounded gaps across {len(per_theme_gaps)} themes "
            f"using '{reduce_model}'..."
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
            print(f"[REDUCE] Error: {e}")

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
        "# Ranked Research Gaps Report",
        "",
        f"- **Map Model:** `{model_name}`",
        f"- **Reduce Model:** `{reduce_model}`",
        f"- **Themes analyzed:** {len(grouped_papers)}",
        f"- **Grounded gaps before reduce:** {total_grounded}",
        f"- **Final ranked gaps:** {len(final_ranked)}",
        "",
        "## Ranked Research Gaps",
        "",
    ]

    for gap in final_ranked:
        themes_str = ", ".join(gap["themes"]) if gap["themes"] else "General"
        refs_str = ", ".join(f"`{rid}`" for rid in gap["supporting_ids"])
        md_lines.extend(
            [
                f"### {gap['rank']}. {gap['title']}",
                f"- **Category:** `{gap['category']}`",
                f"- **Themes:** {themes_str}",
                f"- **Supporting Papers (ID / DOI):** {refs_str}",
                f"- **Rationale:** {gap['rationale']}",
                "",
            ]
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved JSON to {output_json} and Markdown report to {output_md}")
    return result_payload


if __name__ == "__main__":
    run_gap_agent(
        input_file="evaluated_dataset.json",
        output_json="research_gaps_ranked.json",
        output_md="research_gaps_report.md",
        model_name="gemma3:4b",
        reduce_model_name=None,
        num_ctx=4096,
        batch_size=20,
    )