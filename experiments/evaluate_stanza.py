import stanza
from conllu import parse_incr
import tqdm
import argparse
import os

# Ensure stanza is downloaded
stanza.download('la')

def evaluate_stanza(test_file):
    # Map test file to stanza package
    package = 'ittb'
    if 'perseus' in test_file.lower():
        package = 'perseus'
    elif 'proiel' in test_file.lower():
        package = 'proiel'
    elif 'llct' in test_file.lower():
        package = 'llct'
    elif 'udante' in test_file.lower():
        package = 'udante'
        
    print(f"Loading Stanza Latin model with package: {package}...")
    # Explicitly set lemma_package to avoid fallback to ittb
    nlp = stanza.Pipeline('la', processors='tokenize,lemma', package=package, lemma_package=package, tokenize_pretokenized=True)
    
    total_tokens = 0
    correct_tokens = 0
    
    print(f"Evaluating Stanza on {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        sentences = list(parse_incr(f))
        
    for sentence in tqdm.tqdm(sentences):
        # Extract pre-tokenized forms
        tokens = [token["form"] for token in sentence if isinstance(token["id"], int)]
        gold_lemmas = [token["lemma"] for token in sentence if isinstance(token["id"], int)]
        
        # Stanza expects a list of words for pre-tokenized input
        doc = nlp([tokens])
        
        pred_lemmas = []
        for sent in doc.sentences:
            for word in sent.words:
                pred_lemmas.append(word.lemma or "_")
        
        # Match lengths
        if len(pred_lemmas) != len(gold_lemmas):
             # This shouldn't happen with pre-tokenized=True but let's be safe
             for i in range(min(len(gold_lemmas), len(pred_lemmas))):
                total_tokens += 1
                if gold_lemmas[i].lower() == pred_lemmas[i].lower():
                    correct_tokens += 1
        else:
            for i, (gold, pred) in enumerate(zip(gold_lemmas, pred_lemmas)):
                total_tokens += 1
                if gold.lower() == pred.lower():
                    correct_tokens += 1
                elif total_tokens < 100:
                    print(f"Error: Word '{tokens[i]}' | Gold: '{gold}' | Pred: '{pred}'")

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"\nStanza Accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_stanza(args.test_file)
