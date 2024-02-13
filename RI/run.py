import argparse
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import numpy as np
# import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs_mimic
from model import result_identification, pure_bert, result_identification_reverse
# from model_new_ablation import MIMIC_Bert_GAT_init,MIMIC_Bert_Only

from trainer import cross_val,eval_only
import time

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='biomarker_data/dependency/',
                        help='data folder.')
    parser.add_argument('--output_dir', type=str, default='output/mimic_1/',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes.')

    parser.add_argument('--cuda_id', type=str, default='cuda',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')
    parser.add_argument('--bert_model', type=str, default='../clinicalBERT-master/model/pretraining/',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')
    parser.add_argument('--dependency_dim', type=int, default=300,
                        help='Dimension of dep embeddings')
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--cross_val_fold', default=5, type=int, help='k-fold cross validation')
    parser.add_argument('--dataset_name', type=str, default='cancer',
                        help='Choose dataset.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--mode', type=str, default='bert',
                        help='Choose dataset.')
    parser.add_argument('--load_classification_path', type=str, default='../clinicalBERT-master/model/pretraining/',
                        help='classification model path')
    parser.add_argument('--load_model', action='store_true',
                        help='training')
    return parser.parse_args()

def check_args(args):
    '''
    eliminate confilct situations

    '''
    logger.info(vars(args))



def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = parse_args()
    check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    # device = torch.device('cpu')
    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')
    args.device = device

    set_seed(args)

    if args.load_model:
        tokenizer = BertTokenizer.from_pretrained(args.load_classification_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    args.tokenizer = tokenizer

    train_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs_mimic(args)
    if args.mode == 'bert':
            model = pure_bert(args)
    else:
        # model =result_identification(args, 2, 4)
        model =result_identification(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
        # model =result_identification_reverse(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    model.to(args.device)
    if args.load_model:
        checkpoint = torch.load(args.load_classification_path+'model.pth', map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        eval_only(args, train_dataset, model)
    else:
        cross_val(args, train_dataset, model)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))