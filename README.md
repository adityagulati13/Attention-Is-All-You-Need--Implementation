Neural Machine Translation (EN→IT) Transformer — Train, Translate & Visualize Attention

A from-scratch PyTorch Transformer (Encoder–Decoder) for English→Italian translation on Helsinki-NLP/opus_books with:

Word-level tokenization (tokenizers)

Training with early stopping + ReduceLROnPlateau

Validation metrics (BLEU, WER, CER) via torchmetrics

Translation CLI using only translate.py

Attention visualizations exported as interactive HTML (Altair)

📄 Results document: View Results (replace this link)

Table of Contents

Project Structure

Setup

Configuration

Training

Translation (translate.py)

Attention Visualizations

TensorBoard

Troubleshooting & Notes

License

Project Structure
.
├── attention_visual.py       # Exports attention heatmaps (encoder/decoder/cross) as HTML
├── config.py                 # Hyperparams, paths, helpers
├── dataset.py                # BilingualDataset + causal mask
├── model.py                  # Transformer (emb, MHA, FFN, encoder/decoder, projection)
├── train_es_lr.py            # Training loop (early stop + ReduceLROnPlateau + metrics)
├── translate.py              # >>> Translation entrypoint (ONLY this is needed for translate)
├── requirements.txt
└── README.md                 # ← this file

Setup
# 1) Create & activate venv
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt


If you need GPU, install the correct PyTorch wheel for your CUDA version from https://pytorch.org
.

Configuration

Edit config.py as needed:

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "data_source": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


Checkpoints are saved under ./Helsinki-NLP/opus_books_weights/.

Training
python train_es_lr.py


Builds/reuses word-level tokenizers (tokenizer_en.json, tokenizer_it.json)

Trains on opus_books with 90/10 train/val

Saves per-epoch weights and a rolling best model:
Helsinki-NLP/opus_books_weights/tmodel_best.pt

Metrics (BLEU, WER, CER) are logged to TensorBoard.

Translation (translate.py)

translate.py is the only script you need to run translations.

Translate raw text
python translate.py "I am not a very good student."

Translate by dataset index

Passing an integer translates the training-split example at that index and prints context:

python translate.py 42

Use from Python
from translate import translate

print(translate("The book is on the table."))
print(translate(123))  # translate the 123rd training example


Notes

The script loads the latest/best checkpoint via latest_weights_file_path(config).

Outputs are post-processed to clean extra spaces before punctuation.

Attention Visualizations

Generate interactive Altair heatmaps for encoder self-attn, decoder self-attn, and cross-attn:

python attention_visual.py


This creates:

encoder_attention.html

decoder_attention.html

cross_attention.html

Ensure the script points to a valid weights file. Recommended:

state = torch.load(latest_weights_file_path(get_config()), map_location=device)

TensorBoard
tensorboard --logdir runs


View loss curves and validation metrics (BLEU/WER/CER).

Troubleshooting & Notes

CUDA build mismatch: install a PyTorch build matching your CUDA; or use CPU-only wheels.

Config typos: use config['data_source'] (not datasource) consistently.

Tokenizer files: if missing, they’ll be created on first training run.

Masks: custom attention uses boolean/binary masks shaped to broadcast with MHA; keep causal_mask() as provided.

License

Educational/research use. Add an explicit license (e.g., MIT) if redistributing.

Results

📄 Attach & update this link:
➡️ Results: BLEU/WER/CER, qualitative samples, screenshots
