from transformers import AutoTokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='./datas/sample_jpn.txt')
parser.add_argument('--language', type=str, choices=['jpn', 'eng'])
args = parser.parse_args()

tokenizer_model = {'eng': "bert-base-uncased", 
                   'jpn': "cl-tohoku/bert-base-japanese"}

file_path = args.file_path
language = args.language

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

texts = []
for line in lines:
    text = line.replace('\n', '')
    texts.append(text)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model[language], clean_up_tokenization_spaces=False)
tokens_list = [tokenizer.tokenize(text) for text in texts]

token_to_id = {' ': 0, 'EOS': 1}
corpus_list = []
for tokens in tokens_list:
    for token in tokens:
        if token not in token_to_id:
            new_id = len(token_to_id)
            token_to_id[token] = new_id
    corpus = [token_to_id[tok] for tok in tokens]
    corpus.append(token_to_id['EOS']) # add EOS to end of sentence.
    corpus_list.append(corpus)

with open('assets.yaml', 'a') as f:
    f.write(f'{language}_token_to_id: {token_to_id}\n')
corpus = np.concatenate(corpus_list, axis=0)
np.save(file_path.replace('.txt', '.npy'), corpus)
