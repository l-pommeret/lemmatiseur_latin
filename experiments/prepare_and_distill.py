import os
import json
import concurrent.futures
import threading
from bs4 import BeautifulSoup
import stanza
import google.generativeai as genai
from collections import defaultdict
import re

# Configure API
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-3-flash-preview"

# Reuse the prompt that worked best
PROMPT_TEMPLATE = """
Role: You are a distinguished Latin philologist and expert in Classical Latin morphology.
Task: Lemmatize the following sequence of Latin words which form a coherent sentence.

Input Words: {tokens_json}

Instructions:
- Return a JSON list of lemmas.
- Ensure every word is mapped to its dictionary citation form (nominative singular for nouns, first person singular present active indicative for verbs).
- Use the sentence context to resolve ambiguities (e.g. 'cum' as preposition vs conjunction).
- If a token is punctuation, return the punctuation mark itself as the lemma.

Response Format: STRICTLY a JSON list of strings. No markdown formatting.
"""

def clean_html(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as f: # Latin Library often uses ISO-8859-1
        soup = BeautifulSoup(f, 'html.parser')
    
    # Extract text from p tags usually containing the body
    text_content = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()
        if text:
            text_content.append(text)
            
    full_text = "\n".join(text_content)
    # Basic cleanup
    full_text = re.sub(r'\s+', ' ', full_text)
    return full_text

def tokenize_text(text):
    print("Tokenizing raw text with Stanza...")
    # Initialize Stanza just for tokenization
    nlp = stanza.Pipeline(lang='la', processors='tokenize', verbose=False)
    doc = nlp(text)
    
    sentences = []
    for sent in doc.sentences:
        tokens = [word.text for word in sent.words]
        sentences.append(tokens)
    
    print(f"Extracted {len(sentences)} sentences.")
    return sentences

# Optimized Batch Processing
BATCH_SIZE = 20  # Number of sentences per API call

BATCH_PROMPT_TEMPLATE = """
Role: You are a distinguished Latin philologist.
Task: Lemmatize the following batches of Latin sentences.

Input: A JSON list of sentences, where each sentence is a list of words.
{sentences_json}

Instructions:
- Return a JSON list of lists of lemmas.
- The outer list must match the number of input sentences.
- Each inner list must match the number of words in the corresponding sentence.
- Use context to resolve ambiguities.

Response Format: STRICTLY a JSON list of lists of strings. No markdown.
"""

def get_batch_lemmas(sentences_batch):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = BATCH_PROMPT_TEMPLATE.format(sentences_json=json.dumps(sentences_batch))
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        batch_lemmas = json.loads(text_response)
        
        # Basic Validation
        if len(batch_lemmas) != len(sentences_batch):
            print(f"Batch length mismatch: {len(sentences_batch)} vs {len(batch_lemmas)}")
            return None
            
        return batch_lemmas
    except Exception as e:
        print(f"Batch API Error: {e}")
        return None

def process_batch(sentences):
    # Split into batches
    batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
    print(f"Processing {len(sentences)} sentences in {len(batches)} batches (Batch Size: {BATCH_SIZE})...")
    
    all_results = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_batch_idx = {
            executor.submit(get_batch_lemmas, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in concurrent.futures.as_completed(future_to_batch_idx):
            batch_idx = future_to_batch_idx[future]
            batch_result = future.result()
            
            if batch_result:
                # Place results in the correct slots
                start_global_idx = batch_idx * BATCH_SIZE
                for local_i, lemmas in enumerate(batch_result):
                    if start_global_idx + local_i < len(all_results):
                        all_results[start_global_idx + local_i] = lemmas
            
            print(f"Batch {batch_idx+1}/{len(batches)} finished.")
                
    return all_results

def save_conllu(sentences, all_lemmas, output_file):
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent_idx, (tokens, lemmas) in enumerate(zip(sentences, all_lemmas)):
            if lemmas is None:
                continue # Skip failed generations
            
            f.write(f"# sent_id = silver_cicero_{sent_idx+1}\n")
            f.write(f"# text = {' '.join(tokens)}\n")
            
            for i, (word, lemma) in enumerate(zip(tokens, lemmas)):
                # CoNLL-U format: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
                # We only have FORM and LEMMA. Others will be underscore.
                line = f"{i+1}\t{word}\t{lemma}\t_\t_\t_\t_\t_\t_\t_\n"
                f.write(line)
            
            f.write("\n")

def main():
    raw_file = "data/raw_text/cicero_in_catilinam.txt"
    output_file = "data/silver_cicero.conllu"
    
    # 1. Get Text
    if not os.path.exists(raw_file):
        print(f"File not found: {raw_file}")
        return
        
    text = clean_html(raw_file)
    
    # 2. Tokenize
    sentences = tokenize_text(text)
    
    # 3. Distill (Annotate)
    print("Starting Distillation with Gemini...")
    all_lemmas = process_batch(sentences)
    
    # 4. Save
    save_conllu(sentences, all_lemmas, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
