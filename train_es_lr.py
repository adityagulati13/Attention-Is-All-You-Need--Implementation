import torch.cuda
from datasets import load_dataset
import torch.nn as nn
from model import build_transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_config, get_weights_file_path, latest_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchmetrics
import warnings
import os


# validation code
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # autoregressive inference logic
    # will generate a target sequence, one token at a time
    # purpose-> generates output translation from a source sequence using Transformer model
    # getting index of start of sent and end of sent
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    # encoder output->runs once to to get contextual representation of input sentence
    encoder_output = model.encode(source, source_mask)
    # adding sos token to decoder input (autoregression-->right shift operation)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(
        device)  # tensor ready to be passed as input to the decoder
    # loop until decoder reaches the max_len or [EOS] is encountered
    while True:
        if decoder_input.size(1) == max_len:
            break
            # mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # next token
        prob = model.project(
            out[:, -1])  # generate(last decoder ts output)->project to vocb space->prob(logits for next token pred)
        _, next_word = torch.max(prob, dim=1)  # next_word-->max_prob
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )  # add this next_word to decoder output
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


# model evaluation

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=None):  # default: use entire validation set
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1  # validation batch size must be 1 for greedy decode
            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f"{f'source: ':>12}{source_text}")
            print_msg(f"{f'target: ':>12}{target_text}")
            print_msg(f"{f'predicted: ':>12}{model_out_text}")

            # Only stop early if a cap was explicitly provided
            if num_examples is not None and count >= num_examples:
                print_msg('-' * console_width)
                break


    if writer:
        # evaluate char error rate
        # for character level mistakes
        cer_metric = torchmetrics.CharErrorRate()
        cer = cer_metric(predicted, expected)
        writer.add_scalar('validation/CER', cer, global_step)
        writer.flush()

    # word error rate
    # for word level mistakes

    wer_metric = torchmetrics.WordErrorRate()

    wer = wer_metric(predicted, expected)
    writer.add_scalar('validation/WER', wer, global_step)
    writer.flush()

    # bleu score
    # comparison of n-gram between predicted and reference sent
    bleu_metric = torchmetrics.BLEUScore()
    expected_bleu = [[t] for t in expected]  # nested
    bleu = bleu_metric(predicted, expected_bleu)
    writer.add_scalar('validation/BLEU', bleu, global_step)
    writer.flush()




def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]  # hf ds --> dict , get a specific sent in specific lang


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))  # path to tokenizer
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # New wordlevel tokenizer with ["UNK"]
        tokenizer.pre_tokenizer = Whitespace()  # splitting criteria
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                   min_frequency=2)  # setting up the trainer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),
                                      trainer=trainer)  # training a tokenizer from an iterable source of data
        # proceses each sent->split it into tokens-> counts word freq->Builds vocab-> TRAINS TOKENIZER
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# Tokenizer-->a Class
# tokenizer--> an instance or object

# config--> {datasource, lang_src, lang_tgt, tokenizer_file}
def get_ds(config):
    ds_raw = load_dataset(f"{config['data_source']}", f"{config['lang_src']}-{config['lang_tgt']}",
                          split='train')  # loading ds from hf (only train data -> further spit to train and val)

    # building tokeinzer # ds-> raw_ds
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    # train test split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # getting the dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])
    # max length of each sent in the source and target sentence

    max_len_src = 0
    max_len_tgt = 0
    # iterate through the orignal dataset to compute maximun tokenized len

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source sentence: {max_len_src}')  # largest seq in src
    print(f'Max length of target sentence: {max_len_tgt}')  # largest in tgt
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"],
                              d_model=config['d_model'])
    return model


def compute_val_loss(model, val_dataloader, tokenizer_tgt, device):
    model.eval()
    total_loss = 0.0
    count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'))

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / count


def train_model(config):
    import time
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(f"{config['data_source']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config,
                                                                                                        preload) if preload else None

    if os.path.exists(model_filename):
        print(f'Resuming from best model: {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        if initial_epoch >= config['num_epochs']:
            config['num_epochs'] = initial_epoch + 3
        global_step = state['global_step']
        best_val_loss = compute_val_loss(model, val_dataloader, tokenizer_tgt, device)
        print(f"Resumed with best validation loss: {best_val_loss:.4f}")
    else:
        print("Training from scratch")
        best_val_loss = float('inf')

    no_improve_count = 0
    patience = 10

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        val_loss = compute_val_loss(model, val_dataloader, tokenizer_tgt, device)
        writer.add_scalar("val loss", val_loss, global_step)
        writer.flush()
        print(f"Validation loss at epoch {epoch:02d}: {val_loss:.4f}")

        #Update LR scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            print(f"New best model at epoch {epoch:02d}, saving as best...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, get_weights_file_path(config, "best"))
        else:
            no_improve_count += 1
            print(f"No improvement ({no_improve_count}/{patience})")

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        if no_improve_count >= patience:
            print(f" Early stopping triggered at epoch {epoch:02d}. Best val loss: {best_val_loss:.4f}")
            break

    total_time = (time.time() - start_time) / 60
    print(f"Training complete in {total_time:.2f} minutes. Best val loss: {best_val_loss:.4f}")



if __name__ == '__main__':
    import time

    start_time = time.time()
    warnings.filterwarnings("ignore")
    config = get_config()
    ####
    config['preload'] = "best"
    train_model(config)
    end_time = time.time()
    print(f" Total Training Time: {(end_time - start_time) / 60:.2f} minutes")
