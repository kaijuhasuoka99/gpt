import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gpt.gpt import GPT, GPTConfig
import yaml
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./sample_jpn.npy')
parser.add_argument('--save_weights_folder', type=str, default='./weights/')
parser.add_argument('--language', type=str, choices=['jpn', 'eng'])
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=2.0e-4)
parser.add_argument('--grad_norm_clip', type=int, default=1.0)
args = parser.parse_args()

data_path = args.data_path
save_weights_folder = args.save_weights_folder + args.language + '/'
language = args.language
epochs = args.epochs
batch = args.batch
lr = args.lr
grad_norm_clip = args.grad_norm_clip

class Dataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        sample = self.datas[idx]
        input = sample[:-1]
        target = sample[1:]
        x = torch.tensor(input, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)

        return x, y

config = GPTConfig()
seq_len = config.seq_len
device = config.device

corpus = np.load(data_path)
datas = []
for i in range(len(corpus) - (seq_len+1)):
    datas.append(corpus[i:i+(seq_len+1)])
datas = np.array(datas)

with open('assets.yaml', 'r') as f:
    assets = yaml.safe_load(f)
token_to_id = assets[f'{language}_token_to_id']

# for i in range(datas.shape[0]):
#     zero_index = np.where(datas[i] == token_to_id['EOS'])[0]
#     if zero_index.size > 0:
#         zero_index = zero_index[0]
#         datas[i, zero_index:] = token_to_id[' ']

config.n_vocab = len(token_to_id)
gpt = GPT(config).to(device)
dataset = Dataset(datas)

optimizer = torch.optim.Adam(gpt.parameters(), lr=lr)

def run_epoch():
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch)
    
    epoch_loss = []
    with tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = gpt(x)
            loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1), ignore_index=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), grad_norm_clip)
            optimizer.step()

            epoch_loss.append(loss.item())

            pbar.set_description(f'epoch: {epoch:3d} loss: {loss:.4f}')
            pbar.update(1)

    return sum(epoch_loss) / len(epoch_loss)

for epoch in range(1, epochs+1):
    gpt.train()
    loss = run_epoch()
    tqdm.write(f'mean loss: {loss:.4f}') 

    if epoch > 1:
        if loss > prev_loss:
            break
    prev_loss = loss

    if not os.path.exists(save_weights_folder + str(epoch)):
        os.makedirs(save_weights_folder + str(epoch))
    torch.save(gpt.state_dict(), save_weights_folder + str(epoch) + '/gpt.pth')

        
    