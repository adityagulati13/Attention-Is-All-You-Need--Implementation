
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer
from datasets import load_dataset

from config import get_config, latest_weights_file_path
from model import build_transformer
from dataset import BilingualDataset, causal_mask  # uses (1,1,L,L) causal mask

# defining entry point that would either translate a raw string or int index
def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    config = get_config()

    # Load--> wordlevel tokenizer both src and tgt
    tok_src = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_src"]))))
    tok_tgt = Tokenizer.from_file(str(Path(config["tokenizer_file"].format(config["lang_tgt"]))))

    # model building
    model = build_transformer(
        tok_src.get_vocab_size(),
        tok_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # loading the wts
    ckpt_path = latest_weights_file_path(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # numeric input to index ( from the (train) set)
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)
        ds_raw = load_dataset(
            config["data_source"], f"{config['lang_src']}-{config['lang_tgt']}", split="train"
        )
        ds = BilingualDataset(
            ds_raw, tok_src, tok_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"]
        )
        sentence = ds[idx]["src_text"]
        label = ds[idx]["tgt_text"]

    seq_len = config["seq_len"]

    # Encode source
    with torch.no_grad():
        src_ids = tok_src.encode(sentence).ids[: seq_len - 2]  # reserve [SOS],[EOS]
        pad_src = seq_len - len(src_ids) - 2
        if pad_src < 0:
            pad_src = 0

        sos_src = tok_src.token_to_id("[SOS]")
        eos_src = tok_src.token_to_id("[EOS]")
        pad_src_id = tok_src.token_to_id("[PAD]")

        source = torch.tensor(
            [sos_src] + src_ids + [eos_src] + [pad_src_id] * pad_src, dtype=torch.long, device=device
        )  # (L,)
        source = source.unsqueeze(0)  # (B=1, L)

        # src mask -> (B,1,1,L)
        src_mask = (source != pad_src_id).unsqueeze(1).unsqueeze(2).int()  # (1,1,1,L)

        # Run encoder
        enc_out = model.encode(source, src_mask)

        # Greedy decode (with tiny min-length )
        sos_tgt = tok_tgt.token_to_id("[SOS]")
        eos_tgt = tok_tgt.token_to_id("[EOS]")

        dec = torch.tensor([[sos_tgt]], dtype=torch.long, device=device)  # (1,1)
        min_len = 2  # block EOS for first couple tokens
        steps = 0

        if label != "":
            print(f"{'ID:':>12} {idx}")
        print(f"{'SOURCE:':>12} {sentence}")
        if label != "":
            print(f"{'TARGET:':>12} {label}")
        print(f"{'PREDICTED:':>12}", end=" ")

        while dec.size(1) < seq_len:
            L = dec.size(1)
            tgt_mask = causal_mask(L).to(device)  # (1,1,L,L), keep=1 mask

            out = model.decode(enc_out, src_mask, dec, tgt_mask)  # (1,L,d_model)
            logits = model.project(out[:, -1])  # (1, vocab)
            next_id = torch.argmax(logits, dim=-1).item()

            # prevent early EOS
            if steps < min_len and next_id == eos_tgt:
                # pick 2nd best
                top2 = torch.topk(torch.log_softmax(logits, dim=-1), 2, dim=-1).indices[0]
                next_id = top2[1].item()

            dec = torch.cat([dec, torch.tensor([[next_id]], device=device)], dim=1)
            print(tok_tgt.decode([next_id]), end=" ")

            steps += 1
            if next_id == eos_tgt and steps >= min_len:
                break

        # Trim [SOS]/[EOS] -->detokenize nicely
        out_ids = dec[0].tolist()
        try:
            stop = out_ids.index(eos_tgt)
        except ValueError:
            stop = len(out_ids)
        trimmed = out_ids[1:stop]  # drop SOS and everything after EOS

        text = tok_tgt.decode(trimmed)
        # simple cleanup of spaces before punctuation
        for bad, good in [(" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"), (" ;", ";"), (" :", ":")]:
            text = text.replace(bad, good)

        return text

if __name__ == "__main__":
    # Usage:
    #   PYTHONPATH=. python tf/translate.py "hello world"
    #   PYTHONPATH=. python tf/translate.py 42
    arg = sys.argv[1] if len(sys.argv) > 1 else "I am not a very good student."
    arg = int(arg) if arg.isdigit() else arg
    out = translate(arg)
    print("\nFINAL:", out)
