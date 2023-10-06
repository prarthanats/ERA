import torch
from transformer import Transformer
from transformers import AutoTokenizer
from GPT_Utils import encode, decode,get_batch,estimate_loss
from config import get_gpt_config

config = get_gpt_config()
NUM_EMBED = config['NUM_HEAD'] * 128


def load_and_tokenize_data(path_to_data):
    data_raw = open(path_to_data, encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size,tokenizer

def initialize_model(vocab_size,NUM_EMBED):
    model = Transformer(
        vocab_size=vocab_size,
        num_embed=NUM_EMBED,
        block_size=config['BLOCK_SIZE'],
        num_heads=config['NUM_HEAD'],
        num_layers=config['NUM_LAYER'],
        dropout=config['DROPOUT'],
    )
    m = model.to(config['DEVICE'])
    optimizer = torch.optim.AdamW(m.parameters(), lr=config['LEARNING_RATE'])
    return m, optimizer

def train_model_gpt(model, optimizer, train_data, val_data):
    for step in range(config['MAX_ITER']):
        if step % config['EVAL_INTER'] == 0 or step == config['MAX_ITER'] - 1:
            loss_train = estimate_loss(
                data=train_data, model=model, block_size=config['BLOCK_SIZE'], batch_size=config['BATCH_SIZE']
            )
            loss_val = estimate_loss(
                data=val_data, model=model, block_size=config['BLOCK_SIZE'], batch_size=config['BATCH_SIZE']
            )
            print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

        xb, yb = get_batch(data=train_data, block_size=config['BLOCK_SIZE'], batch_size=config['BATCH_SIZE'])
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def generate_output(model, tokenizer, context):
    generated_sequence = model.generate(idx=context, max_new_tokens=100, block_size=config['BLOCK_SIZE'])[0]
    decoded_sequence = decode(enc_sec=generated_sequence, tokenizer=tokenizer)
    return decoded_sequence
