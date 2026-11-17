import json
from pydantic import BaseModel
from ollama import Client

client = Client(host="http://localhost:11434")
# 1. Define the exact schema you want Ollama to return using Pydantic

def _build_prompt(paper: dict) -> str:
    title = paper.get("title") or "Unavailable"
    keywords = paper.get("keywords") or []
    keywords_text = ", ".join(keywords) if keywords else "Unavailable"
    abstract = paper.get("abstract") or "Unavailable"

    return f"""
    You are an expert agricultural entomologist. You are filtering scientific papers for relevance to the pest Cryptoblabes gnidiella (honeydew moth) in the context of grape production, grapevine, and viticulture.

    You will be provided with properties of a scientific paper, including its title, keywords, and abstract, depending on availability.
    Determine whether the paper is genuinely about the biology, control, distribution, or ecological impact of Cryptoblabes gnidiella in the context of grape production, grapevine, and viticulture.

    TITLE:
    {title}

    KEYWORDS:
    {keywords_text}

    ABSTRACT:
    {abstract}

    **CRITICAL** - You must output your final decision as a JSON object wrapped in triple backticks. The JSON must have exactly two keys: "is_relevant" (boolean) and "reason" (string).
    """


def evaluate_papers(input_file="papers_dataset.json", output_file="evaluated_dataset.json", model_name="gemma-fullctx"):
    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} papers. Starting evaluation...")
    
    # Process each paper
    for index, paper in enumerate(dataset):
        # if index >=10: break #for debug
        print(f"[{index + 1}/{len(dataset)}] Evaluating: {paper['title'][:50]}...")
        
        prompt = _build_prompt(paper)
        
        try:
            # Send the prompt to your local Ollama instance
            # By passing our Pydantic model to the format parameter, Ollama guarantees valid JSON matching that schema.
            response = client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0}  # Set temperature to 0 for maximum strictness/determinism
            )
            
            # The output is guaranteed to be a stringified JSON matching our Pydantic model
            
            
            # Parse the JSON string directly into a Python dictionary
            
            
            # Append the results directly to the paper record
            
            paper["prompt_text"] = prompt
            
        except Exception as e:
            print(f"Error evaluating paper {paper.get('id')}: {e}")
            
            paper["prompt_text"] = prompt

    # Save the updated dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\nFinished! Evaluated dataset saved to {output_file}")
    

if __name__ == "__main__":
    # Make sure you have run `ollama run llama3.1` in your terminal first!
    evaluate_papers()