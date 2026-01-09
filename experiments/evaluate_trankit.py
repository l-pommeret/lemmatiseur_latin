from trankit import Pipeline
from conllu import parse_incr
import tqdm
import argparse
import os

def evaluate_trankit(test_file):
    # Mapping to trankit treebank names if needed, but 'latin' is usually a good general start
    # Trankit often uses specific names like 'latin-proiel', 'latin-ittb', etc.
    treebank = 'latin'
    if 'ittb' in test_file.lower():
        treebank = 'latin-ittb'
    elif 'proiel' in test_file.lower():
        treebank = 'latin-proiel'
    elif 'perseus' in test_file.lower():
        treebank = 'latin-perseus'
    elif 'llct' in test_file.lower():
        treebank = 'latin-llct'
    elif 'udante' in test_file.lower():
        # UDante might not be in the older trankit version, but let's try
        treebank = 'latin-udante'
    
    print(f"Loading Trankit model for treebank: {treebank}...")
    try:
        nlp = Pipeline(treebank, embedding='xlm-roberta-base')
    except Exception as e:
        print(f"Could not load specific treebank {treebank}, falling back to general 'latin'")
        nlp = Pipeline('latin', embedding='xlm-roberta-base')

    total_tokens = 0
    correct_tokens = 0
    
    print(f"Evaluating Trankit on {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        sentences = list(parse_incr(f))
        
    for sentence in tqdm.tqdm(sentences):
        # Extract pre-tokenized forms
        tokens = [token["form"] for token in sentence if isinstance(token["id"], int)]
        gold_lemmas = [token["lemma"] for token in sentence if isinstance(token["id"], int)]
        
        # Trankit expects a list of words for pre-tokenized input if we usepos_lemma
        # But for lemmatization only, we can use 'lemmatize'
        # Actually trankit's pipeline might need specific calls.
        # Following their doc for pre-tokenized:
        try:
            res = nlp.lemmatize([tokens])
            pred_lemmas = []
            for s in res['sentences']:
                for w in s['tokens']:
                    pred_lemmas.append(w.get('lemma', '_'))
            
            # Match lengths
            if len(pred_lemmas) != len(gold_lemmas):
                 for i in range(min(len(gold_lemmas), len(pred_lemmas))):
                    total_tokens += 1
                    if gold_lemmas[i].lower() == pred_lemmas[i].lower():
                        correct_tokens += 1
            else:
                for gold, pred in zip(gold_lemmas, pred_lemmas):
                    total_tokens += 1
                    if gold.lower() == pred.lower():
                        correct_tokens += 1
        except Exception as e:
            print(f"Error processing sentence: {e}")
            continue

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"\nTrankit Accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_trankit(args.test_file)
