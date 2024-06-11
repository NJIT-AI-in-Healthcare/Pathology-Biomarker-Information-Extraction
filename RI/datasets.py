from torch.nn.utils.rnn import pad_sequence
import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import torch
from nltk import word_tokenize
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast

from transformers import (BertConfig,
                          BertForTokenClassification,
                          BertTokenizer)
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
import spacy
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)
def save_list(file_path,data):
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)
def get_rolled_and_unrolled_data_mimic_new(input_data, args):
    keeped_words = ['without', 'within', 'w/o', 'against', 'no', 'not', 'nt', "n't", 'against', "aren't", 'but',
                    "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                    "needn't", 'no', 'nor', 'not', 'now', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                    "won't", 'wouldn', "wouldn't", 'after', 'again', 'aren', 'before', 'couldn', 'didn', 'doesn', 'don',
                    'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', "mustn't", 'needn', 'shouldn']
    stop_words = set([w for w in stopwords.words('english') if w not in keeped_words] + ["'s", "s'", "please"])

    exclude_tag = ['NUM', 'INTJ', 'PUNCT', 'X', 'PRON', 'SYM', 'PROPN', 'DET', 'PART', 'CCONJ', 'ADP', 'AUX']
    input_data['processed_note'] = None
    for index, row in tqdm(input_data.iterrows(),desc='get dependency'):
        e = ast.literal_eval(row['dependency'])
        processed_note = []

        e['tokens'] = [x.lower() for x in e['tokens']]
        sent_len = len(e['tokens'])
        # temp_sent.append(sent_len)
        aspects = []
        froms = []
        tos = []
        stripped_dependencies = []

        for tup in e['dependencies']:
            if tup[0] == 'root':
                continue
            if len(e['tokens'][tup[1]-1])==1 or len(e['tokens'][tup[2]-1])==1 or e['tokens'][tup[1]-1] in stop_words or e['tokens'][tup[2]-1] in stop_words or e['tokens'][tup[1]-1] in string.punctuation or e['tokens'][tup[2]-1] in string.punctuation:
                continue

            stripped_dependencies.append(tup)
        e['dependencies'] = stripped_dependencies

        pos_class = e['tags']
        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect'])):
            aspect = e['aspect'][i].lower()
            
            # tokenize the aspect
            aspect = word_tokenize(aspect)

            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            froms.append(frm)
            tos.append(to)

            
            dep_tag, dep_idx, dep_dir, dep_pos, dep_dist = reshape_dependency_tree_new(frm, to, e['dependencies'],multi_hop=True,
                                                                    tokens=e['tokens'],
                                                                    tags=e['tags'],
                                                                    needed_info=[stop_words, exclude_tag,
                                                                                 keeped_words])

            
            processed_note.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect,
                 'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir': dep_dir, 'dep_pos':dep_pos, 'dep_dist':dep_dist,
                 'dependencies': e['dependencies']}
            )
        input_data.at[index, 'processed_note'] = processed_note
    return input_data, processed_note


def reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False,tokens=None, max_hop = 2,tags=None,needed_info=[]):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_pos = []
    dep_idx = []
    dep_dir = []
    dep_dist = []

    stop_words,exclude_tag,keeped_words=needed_info[0],needed_info[1],needed_info[2]
    exclude_tag_wo_noun = [x for x in exclude_tag if x != 'NOUN']

    for i in range(as_start, as_end):
        for dep in dependencies:
            
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_pos.append(tags[dep[2] - 1])
                        dep_dir.append(1)
                        dep_dist.append(1)
                    else:
                        dep_pos.append('<pad>')
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                        dep_dist.append(2)
                    dep_idx.append(dep[2] - 1)
                    
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_pos.append(tags[dep[1] - 1])
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                        dep_dist.append(1)
                    else:
                        dep_pos.append('<pad>')
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                        dep_dist.append(2)
                    dep_idx.append(dep[1] - 1)
                    
    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_pos.append(tags[dep[2] - 1])
                                dep_tag.append(dep[0])
                                dep_dir.append(1)
                                dep_dist.append(1)
                            else:
                                dep_pos.append('<pad>')
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_dist.append(2)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append(dep[0])
                                dep_pos.append(tags[dep[1] - 1])
                                dep_dir.append(2)
                                dep_dist.append(1)
                            else:
                                dep_pos.append('<pad>')
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                                dep_dist.append(2)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_pos.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)
            dep_dist.append(2)
    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_pos = [dep_pos[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]
    dep_dist = [dep_dist[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir, dep_pos, dep_dist

def load_and_cache_vocabs(data, args,pkls_path=None):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    if not pkls_path:
        pkls_path = os.path.join(args.output_dir, 'pkls')
        if not os.path.exists(pkls_path):
            os.makedirs(pkls_path)

    word_vocab = None
    word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab

def _default_unk_index():
    return 1

def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    # itos = ['<pad>', '<unk>']
    itos = ['<pad>']
    min_freq = max(min_freq, 1)
    
    # print('counter',counter)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        # tags = d['tags']
        tags = d['dep_pos']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        # if word == '<pad>':
        #     continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i-1 for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def load_datasets_and_vocabs_mimic(args):
    train = pd.read_csv(args.data_dir + 'data.csv')
    # train = train[2936:2942]
    # test = pd.read_csv(args.dataset_folder + 'test.csv')

    train_processed, train_flatten = get_rolled_and_unrolled_data_mimic_new(train, args)
    # test_processed, test_flatten = get_rolled_and_unrolled_data_mimic_new(test, args)
    if not os.path.exists(args.output_dir + '/processed_data'):
        os.makedirs(args.output_dir + '/processed_data')

    # save_list(args.output_dir + '/processed_data' + '/max_fact.pkl', [train_max_fact, test_max_fact])
    logger.info('****** After unrolling ******')

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    # word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
    #     train_flatten + test_flatten, args)
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_flatten, args)
    

    train_dataset = MIMIC_Depparsed_Dataset(
        train_processed, args, word_vocab, dep_tag_vocab, pos_tag_vocab, train_or_test='train')

    # test_dataset = MIMIC_Depparsed_Dataset(
    #     test_processed, args, word_vocab, dep_tag_vocab, pos_tag_vocab, train_or_test='test')

    return train_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab

class MIMIC_Depparsed_Dataset(Dataset):
    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab, train_or_test=None):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.formatted_features = []
        self.max_seq_length = 512
        self.train_or_test = train_or_test

        
        self.convert_features_bert()

        if train_or_test == 'train':
            save_list(self.args.output_dir + '/processed_data' + '/train_formatted_features.pkl',
                      self.formatted_features)
        elif train_or_test == 'test':
            save_list(self.args.output_dir + '/processed_data' + '/test_formatted_features.pkl',
                      self.formatted_features)


    def __len__(self):
        return len(self.formatted_features)


    def __getitem__(self, idx):
        # e = self.data[idx]
        e = self.formatted_features[idx][0]
        # print(e)
        bert_items = e['input_cat_ids'], e['segment_ids'], e['dep_tag_ids'], e['pos_tag_ids'],e['dep_dist'],e['label']
        # for t in bert_items:
        #     print(t)
        items_tensor = [torch.tensor(t) for t in bert_items]
        return items_tensor+[e['mt_idx'],e['sent_idx'],e['s_id']]

    def convert_features_bert(self):
        """
        BERT features.
        convert sentence to feature.
        """
        nlp = spacy.load("en_core_web_sm")

        for index, row in tqdm(self.data.iterrows(),desc='construct feature'):
            full_note = row['unit']
            sentences = nlp(full_note)
            s = sentences.sents


            sentence_index = []
            _sent_current = 0

            store_token_idx = []
            store_word_indexer = []
            _word_current = 0
            sentence_indexer = []

            pre = 0
            temp_token = []

            for s in sentences.sents:
                temp_indexer = []
                
                words = []
                for word in s:
                    word = word.text
                    words.append(word)
                    word_tokens = self.args.tokenizer.tokenize(word)
                    token_idx = len(temp_token)
                    temp_token.extend(word_tokens)
                    if len(word_tokens) ==0:
                        continue
                    temp_indexer.append(token_idx)
    
                    store_token_idx.extend(word_tokens)
                    if token_idx > self.max_seq_length - 2:
                        break
                    store_word_indexer.append(
                        [_word_current, _word_current + len(word_tokens)])  # start index of the current word
                    _word_current = _word_current + len(word_tokens)
                    if len(store_token_idx) > self.max_seq_length - 2:
                        break
                sentence_indexer.extend(temp_indexer)

            store_token_idx = store_token_idx[:self.max_seq_length - 2]

            tokens_a = store_token_idx

            out = []
            fact = row['processed_note']
            # print('fact',fact)
            fact = fact[0]


            cls_token = "[CLS]"
            sep_token = "[SEP]"
            pad_token = 0
            # tokenizer = self.args.tokenizer

            tokens = []
            word_indexer = []
            aspect_tokens = []
            aspect_indexer = []

            for word in fact['sentence']:
                word_tokens = self.args.tokenizer.tokenize(word)
                token_idx = len(tokens)
                tokens.extend(word_tokens)
                # word_indexer is for indexing after bert, feature back to the length of original length.
                word_indexer.append(token_idx)  # start index of the current word

            # pre_word_indexer = word_indexer

            # aspect
            for word in fact['aspect']:
                word_aspect_tokens = self.args.tokenizer.tokenize(word)
                token_idx = len(aspect_tokens)
                aspect_tokens.extend(word_aspect_tokens)
            tokens = tokens[:self.max_seq_length - 2-len(aspect_tokens)-1]

            tokens = [cls_token] + tokens + [sep_token]
            aspect_tokens = [cls_token] + aspect_tokens + [sep_token]

            input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
            input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
                aspect_tokens)

            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])
            
            mt_idx = [1+len(input_ids)-2+1,1+len(input_ids)-2+1+len(input_aspect_ids[1:])-1]
            sent_idx = [1,1+len(input_ids)-2]
            fact['mt_idx']=mt_idx
            fact['sent_idx']=sent_idx
            

            fact['input_cat_ids'] = input_cat_ids
            fact['segment_ids'] = segment_ids

            fact['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in fact['dep_tag']]

            fact['dep_pos'] = [each_tag if each_tag in self.pos_tag_vocab['stoi'] else '<pad>' for each_tag in fact['dep_pos']]
            fact['pos_tag_ids'] = [self.pos_tag_vocab['stoi'][w] 
                                  for w in fact['dep_pos'] ]
            fact['label'] = int(row['polarity'])+1
            fact['s_id'] = row['s_id']

            # print('sent_idx',sent_idx)

            # print("fact['dep_tag_ids']",fact['dep_tag_ids'])
            # print("fact['pos_tag_ids']",fact['pos_tag_ids'])

            new_dep_tag = []
            new_pos_tag = []
            new_dep_dist = []
            cur_idx = 0
            # print('sentence_indexer',len(sentence_indexer), len(fact['dep_tag_ids']),len(sentence_indexer)==len(fact['dep_tag_ids']))
            for word_idx, token_idx in enumerate(sentence_indexer):
                while cur_idx < token_idx:
                    new_dep_tag.append(fact['dep_tag_ids'][word_idx - 1])
                    new_pos_tag.append(fact['pos_tag_ids'][word_idx - 1])
                    new_dep_dist.append(fact['dep_dist'][word_idx - 1])
                    cur_idx += 1
                # if word_idx<len(fact['dep_tag_ids']):
                new_dep_tag.append(fact['dep_tag_ids'][word_idx])
                new_pos_tag.append(fact['pos_tag_ids'][word_idx])
                new_dep_dist.append(fact['dep_dist'][word_idx])
                cur_idx += 1
            cur_sent_len = len(input_ids)-2
            # print('new_dep_tag',new_dep_tag)
            # print('new_pos_tag',new_pos_tag)
            while len(new_dep_tag) < cur_sent_len:
                new_dep_tag.append(new_dep_tag[-1])
                new_pos_tag.append(new_pos_tag[-1])
                new_dep_dist.append(new_dep_dist[-1])  
            # print('sentence_indexer',sentence_indexer)
            # print('input_ids',input_ids)
            # print("fact['dep_tag_ids']",fact['dep_tag_ids'])
            # print('new_dep_tag',new_dep_tag)
            # print(len(input_ids) == len(new_dep_tag)+2)
            # print('check',len(fact['dep_tag_ids']), len(new_dep_tag),len(fact['dep_tag_ids']) == len(new_dep_tag))
            fact['dep_tag_ids'] = new_dep_tag[:len(input_ids)-2]
            fact['pos_tag_ids'] = new_pos_tag[:len(input_ids)-2]
            fact['dep_dist'] = new_dep_dist[:len(input_ids)-2]
            # print(len(input_ids),len(fact['dep_tag_ids'])+2)
            assert len(input_ids) == len(fact['dep_tag_ids'])+2



            out.append(fact)
            self.formatted_features.append(out)



def my_collate_mimic(batch):
    input_cat_ids, segment_ids, dep_tag, dep_pos, dep_dist, label, mt_idx, sent_idx, s_id = zip(*batch)
    input_cat_ids = pad_sequence(input_cat_ids,batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids,batch_first=True, padding_value=0)
    dep_tag = pad_sequence(dep_tag,batch_first=True, padding_value=0)
    dep_pos = pad_sequence(dep_pos,batch_first=True, padding_value=0)
    dep_dist = pad_sequence(dep_dist,batch_first=True, padding_value=0)
    return input_cat_ids,segment_ids,dep_tag,dep_pos,dep_dist, torch.stack(label),torch.tensor(mt_idx), torch.tensor(sent_idx),s_id





