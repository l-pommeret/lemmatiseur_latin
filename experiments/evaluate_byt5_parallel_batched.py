import torch
import os
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from conllu import parse_incr
import tqdm
import torch.multiprocessing as mp

def evaluate_shard(gpu_id, model_path, sentences, results_dict, batch_size=4):
    device = f"cuda:{gpu_id}"
    print(f"GPU {gpu_id} starting with batch size {batch_size}...")
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    
    # Process in batches
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
        batch_sents = sentences[i:i+batch_size]
        
        batch_tokens = []
        batch_gold_lemmas = []
        batch_inputs = []
        
        for sentence in batch_sents:
            tokens = [token["form"] for token in sentence if isinstance(token["id"], int)]
            gold_lemmas = [token["lemma"] for token in sentence if isinstance(token["id"], int)]
            batch_tokens.append(tokens)
            batch_gold_lemmas.append(gold_lemmas)
            batch_inputs.append("lemmatize: " + " ".join(tokens))
            
        # Tokenize batch
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=256, repetition_penalty=1.5)
            pred_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
        for idx, pred_text in enumerate(pred_texts):
            pred_lemmas = pred_text.split()
            gold_lemmas = batch_gold_lemmas[idx]
            tokens = batch_tokens[idx]
            
            # Use fallback if sequence length mismatch
            if len(pred_lemmas) != len(gold_lemmas):
                 pred_lemmas = []
                 for token_form in tokens:
                     input_t = f"lemmatize: {token_form}"
                     input_i = tokenizer(input_t, return_tensors="pt").input_ids.to(device)
                     with torch.no_grad():
                         output_i = model.generate(input_i, max_length=32, repetition_penalty=1.5)
                         pred_l = tokenizer.decode(output_i[0], skip_special_tokens=True).strip()
                         pred_lemmas.append(pred_l)

            for gold, pred in zip(gold_lemmas, pred_lemmas):
                total_tokens += 1
                if gold.lower() == pred.lower():
                    correct_tokens += 1
                
    results_dict[gpu_id] = (correct_tokens, total_tokens)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Parallel batched evaluation of ByT5 model.")
    parser.add_argument("--model_path", type=str, default="./saved_models/byt5_base_poetry_final", help="Path to the model checkpoint.")
    parser.add_argument("--test_file", type=str, default="data/UD_Latin-Perseus/la_perseus-ud-test.conllu", help="Path to the test conllu file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    test_file = args.test_file
    batch_size = args.batch_size
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return

    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        sentences = list(parse_incr(f))
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPUs found.")
        return
        
    print(f"Parallelizing batched evaluation of {len(sentences)} sentences on {num_gpus} GPUs (Batch Size: {batch_size})...")
    print(f"Model Path: {model_path}")
    
    shards = [sentences[i::num_gpus] for i in range(num_gpus)]
    
    manager = mp.Manager()
    results_dict = manager.dict()
    processes = []
    
    for i in range(num_gpus):
        p = mp.Process(target=evaluate_shard, args=(i, model_path, shards[i], results_dict, batch_size))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    total_correct = sum(v[0] for v in results_dict.values())
    total_tokens = sum(v[1] for v in results_dict.values())
    
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"\nFinal Parallel Batched Accuracy: {accuracy:.4f} ({total_correct}/{total_tokens})")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
