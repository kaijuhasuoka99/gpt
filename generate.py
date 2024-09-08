import torch
from gpt.gpt import GPT, GPTConfig
from transformers import AutoTokenizer

import numpy as np
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_text', type=str, default='')
parser.add_argument('--load_weights_folder', type=str, default='./weights/jpn/3/')
parser.add_argument('--language', type=str, choices=['jpn', 'eng'])
args = parser.parse_args()

tokenizer_model = {'eng': "bert-base-uncased", 
                   'jpn': "cl-tohoku/bert-base-japanese"}

input_text = args.input_text
load_weights_folder = args.load_weights_folder
language = args.language

if input_text == '':
    if language == 'jpn':
        input_text = "私は"
    elif language == 'eng':
        input_text = "I"


def softmax(a):
    c = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

def topkp_sampling(probs, k=5, p=0.1):
    p_i = k
    k_indices = np.zeros((k,), dtype=np.int32)
    p_probs = np.zeros((k,), dtype=np.float32)
    for i in range(k):
        k_indices[i] = np.argmax(probs)
        p_probs[i] = probs[k_indices[i]]
        if p_probs.sum() >= p:
            p_i = i + 1
            break
        probs[k_indices[i]] = -np.inf

    p_probs = p_probs[:p_i]
    k_indices = k_indices[:p_i]
    p_probs = p_probs / p_probs.sum()
    sample = np.random.choice(k_indices, p=p_probs)
    return sample

with open('assets.yaml', 'r') as f:
    assets = yaml.safe_load(f)
token_to_id = assets[f'{language}_token_to_id']

config = GPTConfig()
seq_len = config.seq_len
device = config.device

config.n_vocab = len(token_to_id)
gpt = GPT(config).to(device)
gpt.load_state_dict(torch.load(load_weights_folder + 'gpt.pth', weights_only=True))
gpt.eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model[language], clean_up_tokenization_spaces=False)
input_tokens = tokenizer.tokenize(input_text)
corpus = [token_to_id[tok] for tok in input_tokens]
input = np.array(corpus)

output = corpus
while True:
    y = gpt.infer(input)
    p = softmax(y)
    y = np.random.choice(len(p), p=p)
    # y = topkp_sampling(p, k=5, p=0.3)

    if token_to_id['EOS'] == y:
        break
    output.append(y)
    input = np.concatenate((input, np.array([y])), axis=0)
    if len(input) > seq_len:
        input = input[-seq_len:]

id_to_token = {}
for i, k in enumerate(token_to_id.keys()):
    id_to_token[i] = k

output_tokens = [id_to_token[id] for id in output]

if language == 'eng':
    output_text = ' '.join(output_tokens).replace(' ##', '').replace(' .', '.')
elif language == 'jpn':
    output_text = ''.join(output_tokens).replace('##', '')
print(output_text)



