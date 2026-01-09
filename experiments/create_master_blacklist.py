from conllu import parse_incr
import os

def extract_sentences():
    test_files = [
        "data/UD_Latin-Perseus/la_perseus-ud-test.conllu",
        "data/UD_Latin-ITTB/la_ittb-ud-test.conllu",
        "data/UD_Latin-PROIEL/la_proiel-ud-test.conllu",
        "data/UD_Latin-LLCT/la_llct-ud-test.conllu",
        "data/UD_Latin-UDante/la_udante-ud-test.conllu"
    ]
    
    unique_sentences = set()
    
    for f_path in test_files:
        if not os.path.exists(f_path):
            print(f"Warning: {f_path} not found.")
            continue
        
        print(f"Extracting from {f_path}...")
        with open(f_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                # Extract clean text from the sentence
                text = sentence.metadata.get("text")
                if not text:
                    # Fallback if text metadata is missing
                    words = [token["form"] for token in sentence if isinstance(token["id"], int)]
                    text = " ".join(words)
                
                # Normalize for blacklist comparison (lower, no spaces)
                normalized = text.strip().lower().replace(" ", "")
                unique_sentences.add(normalized)
    
    output_path = "master_test_blacklist.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sorted(list(unique_sentences)):
            f.write(s + "\n")
    
    print(f"Total unique sentences in master blacklist: {len(unique_sentences)}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract_sentences()
