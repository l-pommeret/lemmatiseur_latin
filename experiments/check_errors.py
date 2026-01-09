import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from conllu import parse_incr
import os

def debug_errors(model_path, test_file, num_samples=10, skip_samples=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    with open(test_file, "r", encoding="utf-8") as f:
        sentences = list(parse_incr(f))[skip_samples : skip_samples + num_samples]
        
    for i, sentence in enumerate(sentences):
        tokens = [token["form"] for token in sentence if isinstance(token["id"], int)]
        gold_lemmas = [token["lemma"] for token in sentence if isinstance(token["id"], int)]
        
        input_text = "lemmatize: " + " ".join(tokens)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=256)
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred_lemmas = pred_text.split()
            
        print(f"\n--- Sentence {i} ---")
        print(f"Input Tokens: {' '.join(tokens)}")
        print(f"Gold Lemmas:  {' '.join(gold_lemmas)}")
        print(f"Pred Lemmas:  {' '.join(pred_lemmas)}")
        
        if len(pred_lemmas) != len(gold_lemmas):
            print(f"WARNING: Length mismatch! Gold {len(gold_lemmas)}, Pred {len(pred_lemmas)}")
            # Try token by token
            print("Token-by-token predictions:")
            token_preds = []
            for t in tokens:
                inp = f"lemmatize: {t}"
                inp_ids = tokenizer(inp, return_tensors="pt").input_ids.to(device)
                out_ids = model.generate(inp_ids, max_length=32)
                l = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
                token_preds.append(l)
            print(f"T-T   Lemmas: {' '.join(token_preds)}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Debug errors of a lemmatization model.")
    parser.add_argument("--model_path", type=str, default="./saved_models/byt5_base_poetry_final", help="Path to the model.")
    parser.add_argument("--test_file", type=str, default="data/UD_Latin-Perseus/la_perseus-ud-test.conllu", help="Path to the test file.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to debug.")
    parser.add_argument("--skip_samples", type=int, default=0, help="Number of samples to skip.")
    args = parser.parse_args()
    
    debug_errors(args.model_path, args.test_file, args.num_samples, args.skip_samples)

if __name__ == "__main__":
    main()
