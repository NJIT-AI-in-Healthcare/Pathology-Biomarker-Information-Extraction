import logging
import argparse
import math
import os
import sys
import random
import numpy as np

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from tqdm import tqdm, trange
from datasets import my_collate_mimic
from transformers import AdamW
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss, BCELoss
from findClosestRes import run_rule_based,record_train_test_id
import pandas as pd


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(args, batch):
    inputs = {  'input_cat_ids_batch': batch[0].to(args.device),
                'segment_ids_batch': batch[1].to(args.device),
                'dep_tags_batch': batch[2].to(args.device),
                'dep_pos_batch':batch[3].to(args.device),
                'dep_dist_batch':batch[4].to(args.device),
                'mt_idx': [t.to(args.device) for t in batch[6]],
                'sent_idx': [t.to(args.device) for t in batch[7]],
                }
    # labels = batch[5].type(torch.LongTensor).to(args.device)
    labels = batch[5].to(args.device)
    s_id = batch[8]
    return inputs, labels, s_id

def save(model, optimizer,output_path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_path)
def get_collate_fn(args):
    return my_collate_mimic

def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer


def train(args, train_dataloader, model,val_data_loader):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    collate_fn = get_collate_fn(args)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = get_bert_optimizer(args, model)
    # _params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optimizer(_params, lr=args.learning_rate, weight_decay=0.01)

    max_val_acc = 0
    max_val_f1 = 0
    max_val_epoch = 0
    global_step = 0
    path = None
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for i_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        n_correct, n_total, loss_total = 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()

            inputs, targets, s_id = get_input_from_batch(args, batch)
            outputs = model(**inputs)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, targets)
            loss.backward()
            optimizer.step()
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)
            if global_step % args.logging_steps == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))


        val_acc, val_f1, val_all_results = _evaluate_acc_f1(args,model,val_data_loader)
        logger.info('> val_acc: {:.4f}, val_f1: {:.4f}, val_precision: {:.4f}, val_recall: {:.4f}, val_f1_2: {:.4f}'.format(val_acc, val_f1, val_all_results[0], val_all_results[1], val_all_results[2]))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_epoch = i_epoch

            output_folder = args.output_dir+'/model/'+"checkpoint_{}".format(global_step)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path=output_folder+'/model.pth'
            save(model, optimizer,output_path)
            args.tokenizer.save_pretrained(output_folder)

            logger.info('>> saved: {}'.format(path))
        if val_f1 > max_val_f1:
            max_val_f1 = val_f1

    return output_path

def cross_val(args, train_dataset, model):
    collate_fn = get_collate_fn(args)
    re=get_train_test_id(args,train_dataset)

    all_test_acc, all_test_f1,all_test_precision,all_test_recall,all_test_f1_2 = [], [],[],[],[]
    for fid in range(args.cross_val_fold):
        logger.info('fold : {}'.format(fid))
        logger.info('>' * 100)
        
        trainset=re[fid][0]
        valset=re[fid][1]
        
        train_dataloader = DataLoader(trainset, batch_size=args.train_batch_size,collate_fn=collate_fn, shuffle=True)
        val_dataloader = DataLoader(valset, batch_size=args.eval_batch_size,collate_fn=collate_fn, shuffle=False)
        # record_train_test_id(val_dataloader,fid)
        # print('done')
        
        
        # train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
        # val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False)
        if args.load_model:
            best_model_path = args.load_classification_path+'model.pth'
        else:

            checkpoint = torch.load(args.bert_model+'pytorch_model.bin', map_location=args.device)
            model.bert.load_state_dict(checkpoint,strict=False)

            best_model_path = train(args, train_dataloader, model,val_dataloader)
        best_checkpoint = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(best_checkpoint['model_state_dict'],strict=False)
        # model.load_state_dict(torch.load(best_checkpoint))
        test_acc, test_f1,test_all_results = _evaluate_acc_f1(args,model,val_dataloader)
        all_test_acc.append(test_acc)
        all_test_f1.append(test_f1)
        all_test_precision.append(test_all_results[0])
        all_test_recall.append(test_all_results[1])
        all_test_f1_2.append(test_all_results[2])
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f},test_precision: {:.4f}, test_recall: {:.4f}, test_f1_2: {:.4f}'.format(test_acc, test_f1, test_all_results[0], test_all_results[1], test_all_results[2]))

    mean_test_acc, mean_test_f1, mean_test_precision, mean_test_recall, mean_test_f1_2 = np.mean(all_test_acc), np.mean(all_test_f1), np.mean(all_test_precision), np.mean(all_test_recall), np.mean(all_test_f1_2)
    logger.info('>' * 100)
    logger.info('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}, mean_test_precision: {:.4f}, mean_test_recall: {:.4f}, mean_test_f1_2: {:.4f}'.format(mean_test_acc, mean_test_f1, mean_test_precision, mean_test_recall, mean_test_f1_2))
def eval_only(args, dataset,model):
    collate_fn = get_collate_fn(args)
    all_test_acc, all_test_f1,all_test_precision,all_test_recall,all_test_f1_2 = [], [],[],[],[]
    val_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,collate_fn=collate_fn, shuffle=False)
    test_acc, test_f1,test_all_results = _evaluate_acc_f1(args,model,val_dataloader)
    all_test_acc.append(test_acc)
    all_test_f1.append(test_f1)
    all_test_precision.append(test_all_results[0])
    all_test_recall.append(test_all_results[1])
    all_test_f1_2.append(test_all_results[2])
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f},test_precision: {:.4f}, test_recall: {:.4f}, test_f1_2: {:.4f}'.format(test_acc, test_f1, test_all_results[0], test_all_results[1], test_all_results[2]))

def get_train_test_id(args,trainset):
    # folder_name = '../data/RI/10fold/'
    folder_name = args.data_dir + '/10fold/'
    re=[]

    j_l=[0,2,4,6,8]
    for e_j in j_l:
    # for j in range(10):
        for j in range(e_j,e_j+2):
            file='fold_'+str(j)+'_sent_id.csv'
            id_df=pd.read_csv(folder_name+file)
            s_ids=id_df['sent_id'].values.tolist()
            # print(s_ids)
            train=[]
            test=[]
            for i in range(len(trainset)):
                if i not in s_ids:
                    train.append(trainset[i])
                else:
                    test.append(trainset[i])
            re.append([train,test])
    return re
def _evaluate_acc_f1(args,model,data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    out = []
    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            t_inputs, t_targets, s_id = get_input_from_batch(args, t_batch)
            t_outputs = model(**t_inputs)
            # print('t_outputs',t_outputs)

            # n_correct += (torch.argmax(t_outputs, -1) == torch.argmax(t_targets, -1)).sum().item()
            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            out.append(torch.argmax(t_outputs, -1) == t_targets)
            n_total += len(t_outputs)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

    acc = n_correct / n_total
    # print('out',out)
    # f1 = metrics.f1_score(torch.argmax(t_targets_all, -1).cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
    f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')
    # print(t_targets_all.cpu())
    # print(torch.argmax(t_outputs_all, -1).cpu())

    matrix=np.zeros((3, 3))

    # for i in range(len(torch.argmax(t_targets_all, -1).cpu())):
    for i in range(len(t_targets_all.cpu())):
        a=t_targets_all.cpu()[i]
        # a=torch.argmax(t_targets_all, -1).cpu()[i]
        b=torch.argmax(t_outputs_all, -1).cpu()[i]
        if a==0:
            if b==0:
                matrix[0][0]+=1
            elif b==1:
                matrix[1][0]+=1
            elif b==2:
                matrix[2][0]+=1
        elif a==1:
            if b==0:
                matrix[0][1]+=1
            elif b==1:
                matrix[1][1]+=1
            elif b==2:
                matrix[2][1]+=1
        elif a==2:
            if b==0:
                matrix[0][2]+=1
            elif b==1:
                matrix[1][2]+=1
            elif b==2:
                matrix[2][2]+=1

    precision_manually=[]
    recall_manually=[]
    f1_manually=[]
    for i in range(len(matrix)):
        precision_manually.append(matrix[i][i]/np.sum(matrix[i]))
        recall_manually.append(matrix[i][i]/np.sum(matrix[:,i]))
    for i in range(len(precision_manually)):
        f1_manually.append(2*precision_manually[i]*recall_manually[i]/(precision_manually[i]+recall_manually[i]))
    # macro_precision=np.mean(precision_manually)
    acc_manually=(matrix[0][0]+matrix[1][1]+matrix[2][2])/np.sum(matrix)
    # print('macro_precision')
    logger.info('> manuall_accuracy: {:.4f}, manuall_precision: {:.4f}, manuall_recall: {:.4f}, manuall_f1: {:.4f}'.format(acc_manually,np.mean(precision_manually), np.mean(recall_manually),np.mean(f1_manually)))


    # all_results=metrics.precision_recall_fscore_support(torch.argmax(t_targets_all, -1).cpu(), torch.argmax(t_outputs_all, -1).cpu(),average='macro')
    all_results=metrics.precision_recall_fscore_support(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),average='macro')
    return acc, f1,all_results


