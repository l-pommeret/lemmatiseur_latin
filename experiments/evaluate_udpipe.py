from ufal.udpipe import Model, Pipeline, ProcessingError, Sentence, InputFormat, OutputFormat
import tqdm
import argparse
import os
from conllu import parse_incr

def evaluate_udpipe(model_path, test_file):
    print(f"Loading UDPipe model from {model_path}...")
    model = Model.load(model_path)
    if not model:
        print(f"Cannot load model from {model_path}")
        return
    
    # We want only lemmatization, but we must use a pipeline
    # We use 'none' for tokenizer because we have pre-tokenized data
    pipeline = Pipeline(model, "none", "none", "none", "conllu")
    error = ProcessingError()
    
    total_tokens = 0
    correct_tokens = 0
    
    print(f"Evaluating UDPipe on {test_file}...")
    with open(test_file, "r", encoding="utf-8") as f:
        sentences = list(parse_incr(f))
        
    for sentence in tqdm.tqdm(sentences):
        # UDPipe 1.x handles Sentence objects or CoNLL-U strings
        # For simplicity, we can use the 'conllu' input format if tokenizer is none
        # and we provide the full CoNLL-U text
        conllu_text = ""
        gold_lemmas = []
        for token in sentence:
            if isinstance(token["id"], int):
                # Using 10-column CoNLL-U format
                line = f"{token['id']}\t{token['form']}\t_\t{token['upos']}\t{token['xpos']}\t_\t0\troot\t_\t_"
                conllu_text += line + "\n"
                gold_lemmas.append(token["lemma"])
        conllu_text += "\n"
        
        # We must use InputFormat to parse CoNLL-U and OutputFormat to get result
        input_format = InputFormat.newInputFormat("conllu")
        output_format = OutputFormat.newOutputFormat("conllu")
        
        input_format.setText(conllu_text)
        s = Sentence()
        if input_format.nextSentence(s):
            model.tag(s, model.DEFAULT)
            processed_conllu = output_format.writeSentence(s)
            
            # Parse processed conllu
            processed_sentences = list(parse_incr(processed_conllu.splitlines()))
            if processed_sentences:
                pred_lemmas = [token["lemma"] for token in processed_sentences[0] if isinstance(token["id"], int)]
        
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

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"\nUDPipe Accuracy: {accuracy:.4f} ({correct_tokens}/{total_tokens})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_udpipe(args.model_path, args.test_file)
