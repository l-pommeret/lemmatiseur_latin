import os
import google.generativeai as genai
from conllu import parse_incr
import time
import json
import concurrent.futures
from collections import defaultdict
import threading

# Configure API Key
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-3-flash-preview"

# Define Prompts
PROMPTS = {
    "basic": """
    You are an expert Latin lemmatizer. 
    I will provide a list of Latin words (tokens) from a sentence.
    Your task is to provide the lemma (dictionary headword) for each token.
    
    Rules:
    1. Output strictly a JSON list of strings.
    2. The list must have exactly the same length as the input list.
    3. Maintain the order exactly.
    4. For ambiguous forms, choose the most likely lemma in a classical context.
    5. Treat proper nouns as their nominative singular.
    
    Tokens: {tokens_json}
    
    Output JSON:
    """,
    
    "philologist": """
    Role: You are a distinguished Latin philologist and expert in Classical Latin morphology.
    Task: Lemmatize the following sequence of Latin words which form a coherent sentence.
    
    Input Words: {tokens_json}
    
    Instructions:
    - Return a JSON list of lemmas.
    - Ensure every word is mapped to its dictionary citation form (nominative singular for nouns, first person singular present active indicative for verbs).
    - Use the sentence context to resolve ambiguities (e.g. 'cum' as preposition vs conjunction).
    
    Response Format: STRICTLY a JSON list of strings. No markdown formatting.
    """
}

def get_lemmas(tokens, prompt_template):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = prompt_template.format(tokens_json=json.dumps(tokens))
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        lemmas = json.loads(text_response)
        
        if len(lemmas) != len(tokens):
            return None
        return lemmas
    except Exception as e:
        # print(f"Error ({e})")
        return None

def process_sentence(sent_idx, gold_tokens, prompt_name, prompt_template):
    input_words = [token['form'] for token in gold_tokens]
    gold_lemmas = [token['lemma'] for token in gold_tokens]
    
    pred_lemmas = get_lemmas(input_words, prompt_template)
    
    match_count = 0
    total = len(input_words)
    
    if pred_lemmas:
        for i, pred in enumerate(pred_lemmas):
            if pred.lower() == gold_lemmas[i].lower():
                match_count += 1
    
    return prompt_name, match_count, total

def benchmark_parallel(limit_sentences=50):
    file_path = "data/UD_Latin-Perseus/la_perseus-ud-test.conllu"
    sentences = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, sentence in enumerate(parse_incr(f)):
            if i >= limit_sentences: break
            gold_tokens = [token for token in sentence if isinstance(token['id'], int)]
            if gold_tokens:
                sentences.append((i, gold_tokens))

    print(f"\nBenchmarking {len(sentences)} sentences with {len(PROMPTS)} prompts in parallel...")

    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # We will submit tasks for all prompts * len(sentences)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for sent_idx, gold_tokens in sentences:
            for p_name, p_temp in PROMPTS.items():
                futures.append(executor.submit(process_sentence, sent_idx, gold_tokens, p_name, p_temp))
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            p_name, correct, total = future.result()
            results[p_name]["correct"] += correct
            results[p_name]["total"] += total
            
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(futures)} tasks done.")

    print("\nResults:")
    for p_name, data in results.items():
        if data["total"] > 0:
            acc = data["correct"] / data["total"]
            print(f"Prompt '{p_name}': {acc:.4f} ({data['correct']}/{data['total']})")

if __name__ == "__main__":
    benchmark_parallel(50)
