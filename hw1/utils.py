import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    elif torch.backends.mps.is_available() \
        and torch.backends.mps.is_built() \
        and not force_cpu:
        device = torch.device("mps")
        if status:
            print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )

def get_best_byte_pair(vocab) -> tuple:
    """
    Given a vocab dictionary of words broken up into tokens mapped
    to their frequency, returns the most frequent pair of tokens.
    """
    pair_freq = dict()
    # Iterate through each word in the vocabulary
    for word, freq in vocab.items():
        # Look at the individual tokens in each word
        tokens = word.split()
        for i in range(len(tokens) - 1):
            # Pair consecutive tokens and count their frequency
            token_pair = (tokens[i], tokens[i+1])
            if token_pair not in pair_freq.keys():
                pair_freq[token_pair] = 0
            pair_freq[token_pair] += freq
    
    # Return the most frequent paired token
    best_pair = max(pair_freq, key=pair_freq.get)
    return best_pair

def merge_vocab(v_in, bp: tuple):
    """
    Given a vocab dictionary of words split into subwords and
    their frequencies, replaces subwords with the given byte pair
    and returns the updated frequencies. 
    """
    # Create a regex object of the two tokens to be replaced
    bigram = re.escape(' '.join(bp))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    v_out = dict()
    for word, freq in v_in.items():
        # Replace the bigram with the new byte-pair and store
        w_out = p.sub(''.join(bp), word)
        v_out[w_out] = freq
    return v_out

def get_tokens(vocab):
    """
    From the dictionary of words separated into byte pairs,
    return a set of the most common byte pairs. 
    """
    tokens = set()
    for word, freq in vocab.items():
        subwords = word.split()
        for i in subwords:
            tokens.add(i)
    return tokens

def build_bpe_table(train, vocab_size=1000):
    """
    Implements BPE as described in Sennrich, Haddow, and Birch 2016. 
    They provide helpful psuedocode, and I also used this tutorial:
    https://leimao.github.io/blog/Byte-Pair-Encoding/
    """
    # Count the occurences of each word in the training data
    subword_list, char_list = [], [] #TODO: char_list?
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    # Separate words into characters
                    subword = " ".join(word) 

                    # Append an end-of-word token
                    subword = subword+ " </w>"
                    subword_list.append(subword)

                    # # Also count the unique characters
                    # char_list.extend([i for i in word])
    
    # Build counter dictionaries of subwords
    vocab = Counter(subword_list)

    # Initialize the set of tokens
    token_set = set(("<pad>", "<start>", "<end>", "<unk>"))

    # Merge the most frequent character pair at each iteration
    merge_num = 1
    while True:
        byte_pair = get_best_byte_pair(vocab)
        vocab = merge_vocab(vocab, byte_pair)

        # Add any additional tokens to your token set
        new_tokens = get_tokens(vocab)
        for i in new_tokens:
            token_set.add(i)

        # Always verbose for BPE
        if True:
            print('Merge #{}'.format(merge_num))
            print('Most common pair: {}'.format(byte_pair))
            print('Number of tokens: {}'.format(len(token_set)))
            print('==========')
            merge_num += 1
        
        # Stop merging once we have 1000 tokens
        if len(token_set) >= 1000:
            break
    
    # Sort tokens by character length
    # To encode in BPE, we need to start with the longest tokens
    token_lens = dict()
    for t in token_set:
        # Shorter </w> to a single character and store the length
        t_adjusted = t.replace("</w>", "$")
        token_lens[t] = len(t_adjusted)

    # Perform the sort by token length
    token_lens = dict(sorted(token_lens.items(), 
                        key=lambda item: item[1], 
                        reverse=True))

    # Store only the tokens themselves     
    token_list = list(token_lens.keys())
    
    # Generate the output dictionaries mapping tokens to indices
    tokens_to_index = {t: i for i, t in enumerate(token_list)}
    # Sort dictionary by length of values (dictionaries are ordered since Python 3.6+)
    # Reference: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    # tokens_to_index = dict(sorted(tokens_to_index.items(), 
    #                        key=lambda item: item[1], reverse=True))
    index_to_vocab = {i: t for t, i in tokens_to_index.items()}

    for k, v in tokens_to_index.items():
        print(f"{k} - {v}")
    
    return tokens_to_index, index_to_vocab

def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets
