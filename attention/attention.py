import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emsize = 200 # embedding dimension
d_model = 512 # the number of expected features in the encoder/decoder inputs (default=512)
n_head = 8 # the number of heads in the multiheadattention models (default=8)
num_encoder_layers = 6 # the number of sub-encoder-layers in the encoder (default=6)
num_decoder_layers = 6 # the number of sub-decoder-layers in the decoder (default=6)
dim_feedforward = 2048 # the dimension of the feedforward network model (default=2048)
dropout = 0.1 # the dropout value (default=0.1)
activation = 'relu' # the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu)
custom_encoder = None # custom encoder (default=None)
customer_decoder = None # custom decoder (default=None)
# n_hid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
# n_layers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder

model = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                   dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                                   custom_decoder=customer_decoder).to(device)
# output = model(src, tgt)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)




# num_epochs = 10
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples / batch_size)
# print(total_samples, n_iterations)
#
# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {labels.shape}")