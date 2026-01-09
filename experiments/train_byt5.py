import os
import torch
import shutil
import logging
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
from conllu import parse_incr

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LemmatizationDataset(Dataset):
    def __init__(self, conllu_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = []
        
        logger.info(f"Loading dataset from {conllu_file}...")
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
        logger.info(f"Loaded {len(self.data)} sentences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SaveBestPerseusCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_loss = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # In multi-dataset evaluation, metrics will have keys like 'eval_perseus_loss'
        current_loss = metrics.get("eval_perseus_loss")
        if current_loss is not None and current_loss < self.best_loss:
            logger.info(f"✨ Perseus Loss improved from {self.best_loss:.4f} to {current_loss:.4f}")
            self.best_loss = current_loss
            
            # Define best model path
            best_model_path = os.path.join(self.output_dir, "best_model_perseus")
            
            # Save the model and tokenizer
            kwargs["model"].save_pretrained(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            logger.info(f"💾 Best model saved to {best_model_path}")

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
    
    output_dir = "./saved_models/byt5_final_sota"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=20, 
        per_device_eval_batch_size=20,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no", # Disable default checkpointing
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
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        callbacks=[SaveBestPerseusCallback(output_dir, tokenizer)]
    )
    
    logger.info("Saving strategy: Only when Perseus loss improves.")
    trainer.train()
    
    # Save final model for safety
    final_output = os.path.join(output_dir, "model_final")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"✅ Training complete. Final model saved to {final_output}")

if __name__ == "__main__":
    train()
