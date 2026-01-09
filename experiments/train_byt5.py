import os
import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
from conllu import parse_incr

class LemmatizationDataset(Dataset):
    def __init__(self, conllu_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = []
        
        print(f"Loading dataset from {conllu_file}...")
        with open(conllu_file, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                tokens = [token["form"] for token in sentence if isinstance(token["id"], int)]
                lemmas = [token["lemma"] for token in sentence if isinstance(token["id"], int)]
                
                if len(tokens) > 128: continue
                if len(tokens) == 0: continue
                
                input_text = "lemmatize: " + " ".join(tokens)
                target_text = " ".join(lemmas)
                
                input_enc = tokenizer(input_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
                target_enc = tokenizer(target_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
                
                self.data.append({
                    "input_ids": input_enc.input_ids.squeeze(),
                    "attention_mask": input_enc.attention_mask.squeeze(),
                    "labels": target_enc.input_ids.squeeze()
                })
        print(f"Loaded {len(self.data)} sentences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train():
    model_name = "google/byt5-base"
    tokenizer = ByT5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 1. Final SOTA Dataset (99k sentences)
    train_dataset = LemmatizationDataset("data/merged_final_sota.conllu", tokenizer)
    
    # 2. Multi-Benchmark Evaluation Suite
    eval_datasets = {
        "perseus": LemmatizationDataset("data/UD_Latin-Perseus/la_perseus-ud-test.conllu", tokenizer),
        "ittb": LemmatizationDataset("data/UD_Latin-ITTB/la_ittb-ud-test.conllu", tokenizer),
        "proiel": LemmatizationDataset("data/UD_Latin-PROIEL/la_proiel-ud-test.conllu", tokenizer),
        "llct": LemmatizationDataset("data/UD_Latin-LLCT/la_llct-ud-test.conllu", tokenizer),
        "udante": LemmatizationDataset("data/UD_Latin-UDante/la_udante-ud-test.conllu", tokenizer)
    }
    
    training_args = TrainingArguments(
        output_dir="./saved_models/byt5_final_sota",
        per_device_train_batch_size=20, 
        per_device_eval_batch_size=20,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        logging_steps=100,
        logging_dir="./logs",
        learning_rate=3e-5,
        weight_decay=0.01,
        push_to_hub=False,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets, # Dict for per-dataset logging
    )
    
    print("🚀 Starting Final ByT5 Universal SOTA Training...")
    print("Each benchmark will be logged separately in the training logs.")
    trainer.train()
    
    model.save_pretrained("./saved_models/byt5_final_sota_complete")
    tokenizer.save_pretrained("./saved_models/byt5_final_sota_complete")
    print("✅ Training complete. Model saved to ./saved_models/byt5_final_sota_complete")

if __name__ == "__main__":
    train()
