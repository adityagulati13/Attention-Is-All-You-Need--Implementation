
from pathlib import Path
import torch
from config import get_config, latest_weights_file_path
from train_es_lr import get_model, get_ds, run_validation
from translate import translate
import sys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # Run validation
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                   lambda msg: print(msg), 0, None, num_examples=2)

    # Read input from sys.argv
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = "I am not a very good student."

    # Run translation
    output = translate(input_text)
    print(f"\nFinal Translation Output:\n{output}")

if __name__ == "__main__":
    main()

