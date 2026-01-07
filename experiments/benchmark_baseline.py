import stanza
from conllu import parse_incr
import logging

# Silence Stanza info
logging.getLogger("stanza").setLevel(logging.WARNING)

def benchmark():
    print("Initializing Stanza Pipeline...")
    # Using 'tokenize_pretokenized=True' to trust our reading of the CoNLL-U words.
    # We include 'pos' because lemmatizer depends on it.
    nlp = stanza.Pipeline(lang='la', processors='tokenize,pos,lemma', tokenize_pretokenized=True, verbose=False)
    
    total_tokens = 0
    correct_lemmas = 0
    
    file_path = "data/UD_Latin-Perseus/la_perseus-ud-test.conllu"
    print(f"Benchmarking on {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for sent_idx, sentence in enumerate(parse_incr(f)):
            # Filter for integer IDs (actual words), skipping range IDs (multi-word tokens wrappers)
            gold_tokens = [token for token in sentence if isinstance(token['id'], int)]
            
            if not gold_tokens:
                continue
                
            input_words = [token['form'] for token in gold_tokens]
            gold_lemmas_list = [token['lemma'] for token in gold_tokens]
            
            # Feed to Stanza
            # nlp(List[List[str]])
            doc = nlp([input_words])
            
            # Compare
            try:
                pred_words = doc.sentences[0].words
                
                # Safety check for alignment
                if len(pred_words) != len(gold_tokens):
                    print(f"Warning: Length mismatch in sentence {sent_idx}. Gold: {len(gold_tokens)}, Pred: {len(pred_words)}")
                    # Fallback matching?
                    # With tokenize_pretokenized=True, this should match exactly unless Stanza drops/adds something weird.
                    continue
                
                for i, word in enumerate(pred_words):
                    total_tokens += 1
                    if word.lemma == gold_lemmas_list[i]:
                        correct_lemmas += 1
                    else:
                        # Optional: Print some errors to see what's wrong
                        if total_tokens % 500 == 0:
                            print(f"Error: Form='{word.text}' Gold='{gold_lemmas_list[i]}' Pred='{word.lemma}'")
                            
            except Exception as e:
                print(f"Error processing sentence {sent_idx}: {e}")

    accuracy = correct_lemmas / total_tokens if total_tokens > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct_lemmas}/{total_tokens})")

if __name__ == "__main__":
    benchmark()
