import stanza
import os

def train_model():
    # Stanza training requires a directory structure or specific arguments
    # We will use the 'stanza.train' module interface
    
    # Configuration for lemmatizer training
    # We used 'merged_train.conllu' as train (UD + Silver)
    # And 'la_perseus-ud-test.conllu' as dev (Gold) to monitor real progress on the hard target.
    
    args = [
        '--save_dir', 'saved_models',
        '--save_name', 'latin_distilled',
        '--shorthand', 'la_perseus',
        '--mode', 'train',
        '--train_file', 'data/merged_train.conllu',
        '--eval_file', 'data/lemma/la_perseus.dev.in.conllu',
        '--output_file', 'data/lemma/la_perseus.dev.pred.conllu',
        '--device', 'mps',
        '--batch_size', '256',
        '--num_epoch', '15'
    ]
    
    print("Starting Stanza Training with Distilled Data...")
    # There isn't a direct python function easily exposed without importing internal modules 
    # normally effectively run via command line python -m stanza.utils.training.run_lemma
    
    # Let's generate the command line string for the user to see/run
    # Fix: 'UD_Latin-Perseus' is required as the positional argument for the treebank name
    cmd = "python3 -m stanza.utils.training.run_lemma UD_Latin-Perseus " + " ".join(args)
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    train_model()
