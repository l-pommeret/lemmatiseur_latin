import os
import json
import concurrent.futures
import threading
from bs4 import BeautifulSoup
import stanza
from google import genai
from google.genai import types
from collections import defaultdict
import re
import sys

# Configure API
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

MODEL_NAME = "gemini-3-flash-preview"

# BATCH PROMPT using the 'thinking_v2' logic
BATCH_PROMPT_TEMPLATE = """
Role: You are an expert Latin philologist and the primary architect of the UD Perseus Treebank. 
Task: Lemmatize the following tokens EXACTLY as they are lemmatized in the Universal Dependencies (UD) Perseus test set.

Input: A JSON list of sentences, where each sentence is a list of words.
{sentences_json}

ULTRA-STRICT Rules for UD Perseus Mapping:
1. **Response Format**: STRICTLY a JSON object with a single key "batches", which is a list of lists of strings. Outer list length MUST match input number of sentences. Each inner list length MUST match the corresponding input sentence length.
2. **Adverbs & Connectives**:
   - `aliter` -> `alius`.
   - `primo` -> `primus` (usually).
   - `late` -> `latus`.
   - `quam` -> `qui` (Lemmatize to `qui` unless it is unequivocally a comparative conjunction 'than' or 'as' in a non-relative sense).
   - `neque` for both `nec` and `neque`.
   - `Ne c` or `Nec` -> Map tokens to `ne` and `que` if split, but usually lemmas are `ne` and `que`.
3. **Plurale Tantum & Special Nouns**:
   - `castra` -> `castrum`.
   - `Maenala` -> `Maenala`.
   - `fauces` or `faucibus` -> `fauces`.
   - `mundi` -> `mundum` (in context of "world/universe").
   - `superi` -> `superi`.
   - `Teucri` -> `Teucer`.
4. **Orthography (Classical Deviation)**:
   - Use `auctumnus` (not `autumnus`).
   - Use `inpono`, `inmergo`, `conligo` (NO assimilation).
   - `sepulcrum` (not `sepulchrum`).
   - `Grai` -> `Graii`.
5. **Verb Forms**:
   - `coepimus` -> `coepio`.
   - `frigor` -> `frigor`.
   - Participles: 90% of the time map to the **Verb** (`horreo`, `mando`, `praecipio`, `aperio`, `operio`, `facio`). Only use Adjective for very common fixed adjectives (`tutus`, `diversus`).
6. **Thinking Process**:
   - Resolve `c` after `Nec` as `que`.
   - Verify specific Perseus Treebank headwords for proper names (e.g., `Lavinium` vs `Lavinius`).
"""

def clean_text(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Detect if HTML or TXT
        content = f.read()
        if "<p" in content or "<html" in content:
            soup = BeautifulSoup(content, 'html.parser')
            text_content = [p.get_text().strip() for p in soup.find_all('p')]
            full_text = "\n".join(text_content)
        else:
            full_text = content
            
    full_text = re.sub(r'\[\d+\]', '', full_text)
    full_text = re.sub(r'\d+', '', full_text) # Remove numbers
    full_text = re.sub(r'\s+', ' ', full_text)
    return full_text

def tokenize_text(text, nlp):
    doc = nlp(text)
    sentences = []
    for sent in doc.sentences:
        tokens = [word.text for word in sent.words if word.text]
        if tokens:
            sentences.append(tokens)
    return sentences

def load_blacklist(blacklist_file):
    if not os.path.exists(blacklist_file):
        return set()
    with open(blacklist_file, "r", encoding="utf-8") as f:
        return set(line.strip().lower().replace(" ", "") for line in f)

def is_blacklisted(sentence_tokens, blacklist):
    sentence_str = "".join(sentence_tokens).lower().replace(" ", "")
    return sentence_str in blacklist

BATCH_SIZE = 40

def get_batch_lemmas(sentences_batch):
    client = genai.Client(api_key=GENAI_API_KEY)
    prompt = BATCH_PROMPT_TEMPLATE.format(sentences_json=json.dumps(sentences_batch))
    
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="application/json",
    )
    
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        result = json.loads(response.text)
        
        if isinstance(result, list):
            batch_lemmas = result
        elif isinstance(result, dict):
            batch_lemmas = result.get("batches") or result.get("result") or result.get("lemmas")
            if not batch_lemmas and len(result) == 1:
                batch_lemmas = list(result.values())[0]
        else:
            return None
            
        if not batch_lemmas or len(batch_lemmas) != len(sentences_batch):
            return None
            
        final_batch = []
        for sentence_lemmas in batch_lemmas:
            if not isinstance(sentence_lemmas, list):
                final_batch.append(None)
                continue
            clean_sentence = []
            for l in sentence_lemmas:
                if isinstance(l, dict):
                    lemma_val = l.get("lemma") or l.get("l")
                    clean_sentence.append(str(lemma_val) if lemma_val else str(l))
                else:
                    clean_sentence.append(str(l))
            final_batch.append(clean_sentence)
        return final_batch
    except Exception:
        return None

def main():
    nlp = stanza.Pipeline(lang='la', package='perseus', processors='tokenize', verbose=False)
    blacklist = load_blacklist("master_test_blacklist.txt")
    print(f"Loaded master blacklist with {len(blacklist)} sentences.")
    
    authors = {
        "cicero": "data/lat_text_latin_library/cicero/",
        "seneca": "data/lat_text_latin_library/sen/",
        "livy": "data/lat_text_latin_library/livy/",
        "tacitus": "data/lat_text_latin_library/tacitus/",
        "aquinas": "data/lat_text_latin_library/aquinas/",
        "augustine": "data/lat_text_latin_library/augustine/",
        "vergil": "data/lat_text_latin_library/vergil/",
        "ovid": "data/lat_text_latin_library/ovid/"
    }
    
    output_file = "data/silver_massive.conllu"
    sentences_to_process = []
    prefixes = []
    
    SENTENCES_PER_AUTHOR = 6000 # 6000 * 8 = 48000 sentences
    
    for author, path in authors.items():
        print(f"Collecting data for {author}...")
        count = 0
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
            for f in files:
                if count >= SENTENCES_PER_AUTHOR: break
                text = clean_text(f)
                sentences = tokenize_text(text, nlp)
                for s in sentences:
                    if count >= SENTENCES_PER_AUTHOR: break
                    if not is_blacklisted(s, blacklist) and 3 < len(s) < 100:
                        sentences_to_process.append(s)
                        prefixes.append(author)
                        count += 1
        print(f"Collected {count} sentences for {author}.")

    print(f"Total sentences to distill: {len(sentences_to_process)}")
    
    batches = [sentences_to_process[i:i + BATCH_SIZE] for i in range(0, len(sentences_to_process), BATCH_SIZE)]
    all_lemmas = [None] * len(sentences_to_process)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        future_to_idx = {executor.submit(get_batch_lemmas, batch): i for i, batch in enumerate(batches)}
        for future in concurrent.futures.as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            res = future.result()
            if res:
                start = batch_idx * BATCH_SIZE
                for i, lemmas in enumerate(res):
                    if start + i < len(all_lemmas):
                        all_lemmas[start + i] = lemmas
            print(f"Batch {batch_idx+1}/{len(batches)} done.")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent_idx, (tokens, lemmas, prefix) in enumerate(zip(sentences_to_process, all_lemmas, prefixes)):
            if lemmas is None or len(lemmas) != len(tokens): continue
            f.write(f"# sent_id = massive_{prefix}_{sent_idx+1}\n")
            f.write(f"# text = {' '.join(tokens)}\n")
            for i, (word, lemma) in enumerate(zip(tokens, lemmas)):
                f.write(f"{i+1}\t{word}\t{lemma}\t_\t_\t_\t_\t_\t_\t_\n")
            f.write("\n")
    print("Massive distillation complete.")

if __name__ == "__main__":
    main()
