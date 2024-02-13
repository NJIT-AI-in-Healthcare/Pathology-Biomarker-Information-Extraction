import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from attention import BasicAttention
from torch.nn.utils.rnn import pad_sequence


class result_identification(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(result_identification, self).__init__()
        self.args = args
        self.tokenizer=self.args.tokenizer

        self.bert = BertModel.from_pretrained(args.bert_model)
        config=self.bert.config
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)

        self.type_embed = nn.Embedding(dep_tag_num+1, args.dependency_dim)
        self.pos_embed = nn.Embedding(pos_tag_num+1, args.dependency_dim)

        args.embedding_dim = config.hidden_size

        # self.type_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)
        # self.pos_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)
        # # self.dist_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)

        self.dense = nn.Linear(2*config.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.mlp = nn.Linear(args.dependency_dim, 1)
        self.linear1 = nn.Linear(2*args.embedding_dim, args.embedding_dim)
        self.linear2 = nn.Linear(args.embedding_dim, 1, bias=False)  
        self.mlp2 = nn.Linear(args.embedding_dim, args.embedding_dim)      

    def forward(self, input_cat_ids_batch, segment_ids_batch, dep_tags_batch, dep_pos_batch, dep_dist_batch, mt_idx, sent_idx):
        outputs = self.bert(input_cat_ids_batch, token_type_ids = segment_ids_batch)
        feature_output = outputs[0] # (B, S+A, D)
        pool_out = outputs[1] #(1, S+A, D)
        
        sentence_output = [] # store all sentence embeddings which contain MTs 
        mt_output=[]
        for i in range(feature_output.shape[0]):
            indices = torch.tensor([idx for idx in range(sent_idx[i][0],sent_idx[i][1])]).to(self.args.device) #whether or not +1? index of the sentence in the note
            emb=torch.index_select(feature_output[i], 0, indices) # sentence embedding extracted from the note embedding
            emb=torch.unsqueeze(emb, 0)
            sentence_output.extend(emb)
            
            mt_indices=torch.tensor([idx for idx in range(mt_idx[i][0],min(mt_idx[i][1],feature_output[i].shape[0]))]).to(self.args.device)
            mt_emb=torch.index_select(feature_output[i], 0, mt_indices)
            mt_emb=torch.unsqueeze(mt_emb, 0)
            mt_output.extend(mt_emb)
        sentence_output=pad_sequence(sentence_output,batch_first=True,padding_value=0)
        mt_output=pad_sequence(mt_output,batch_first=True,padding_value=0)
                
        # fmask = (torch.zeros_like(input_cat_ids_batch) == input_cat_ids_batch).float()

        dep_type_embed = self.type_embed(dep_tags_batch)
        pos_type_embed = self.pos_embed(dep_pos_batch)

        combined_embed = torch.stack((dep_type_embed,pos_type_embed),dim=0)
        pooled_embed = torch.max(combined_embed,dim=0)[0]       
        attn_1 = torch.sigmoid(self.mlp(pooled_embed))
        
        sentence_output = sentence_output*attn_1
        
        combined = torch.cat([sentence_output,torch.mean(mt_output,dim=1,keepdim=True).repeat(1,sentence_output.shape[1],1)], dim=2)
        updated_combined = self.linear2(torch.tanh(self.linear1(combined)))
        beta = F.softmax(updated_combined, dim=1)
        sentence_output = sentence_output*beta
        
        x = torch.tanh(self.mlp2(sentence_output))        
        x = torch.mean(x, dim=1)
        x = torch.concat([x,pool_out],dim=-1)
        logit = self.dense(x)

        return logit

class result_identification_reverse(nn.Module):
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(result_identification_reverse, self).__init__()
        self.args = args
        self.tokenizer=self.args.tokenizer

        self.bert = BertModel.from_pretrained(args.bert_model)
        config=self.bert.config
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)

        self.type_embed = nn.Embedding(dep_tag_num+1, args.dependency_dim)
        self.pos_embed = nn.Embedding(pos_tag_num+1, args.dependency_dim)

        args.embedding_dim = config.hidden_size

        # self.type_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)
        # self.pos_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)
        # # self.dist_attn=BasicAttention(args.embedding_dim,config.hidden_size,config.hidden_size)

        self.dense = nn.Linear(2*config.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.mlp = nn.Linear(args.dependency_dim, 1)
        self.linear1 = nn.Linear(2*args.embedding_dim, args.embedding_dim)
        self.linear2 = nn.Linear(args.embedding_dim, 1, bias=False)  
        self.mlp2 = nn.Linear(args.embedding_dim, args.embedding_dim)      

    def forward(self, input_cat_ids_batch, segment_ids_batch, dep_tags_batch, dep_pos_batch, dep_dist_batch, mt_idx, sent_idx):
        outputs = self.bert(input_cat_ids_batch, token_type_ids = segment_ids_batch)
        feature_output = outputs[0] # (B, S+A, D)
        pool_out = outputs[1] #(1, S+A, D)
        
        sentence_output = [] # store all sentence embeddings which contain MTs 
        mt_output=[]
        for i in range(feature_output.shape[0]):
            indices = torch.tensor([idx for idx in range(sent_idx[i][0],sent_idx[i][1])]).to(self.args.device) #whether or not +1? index of the sentence in the note
            emb=torch.index_select(feature_output[i], 0, indices) # sentence embedding extracted from the note embedding
            emb=torch.unsqueeze(emb, 0)
            sentence_output.extend(emb)
            
            mt_indices=torch.tensor([idx for idx in range(mt_idx[i][0],min(mt_idx[i][1],feature_output[i].shape[0]))]).to(self.args.device)
            mt_emb=torch.index_select(feature_output[i], 0, mt_indices)
            mt_emb=torch.unsqueeze(mt_emb, 0)
            mt_output.extend(mt_emb)
        sentence_output=pad_sequence(sentence_output,batch_first=True,padding_value=0)
        mt_output=pad_sequence(mt_output,batch_first=True,padding_value=0)
                
        # fmask = (torch.zeros_like(input_cat_ids_batch) == input_cat_ids_batch).float()

        # dep_type_embed = self.type_embed(dep_tags_batch)
        # pos_type_embed = self.pos_embed(dep_pos_batch)

        # combined_embed = torch.stack((dep_type_embed,pos_type_embed),dim=0)
        # pooled_embed = torch.max(combined_embed,dim=0)[0]       
        # attn_1 = torch.sigmoid(self.mlp(pooled_embed))
        
        # sentence_output = sentence_output*attn_1
        
        combined = torch.cat([sentence_output,torch.mean(mt_output,dim=1,keepdim=True).repeat(1,sentence_output.shape[1],1)], dim=2)
        updated_combined = self.linear2(torch.tanh(self.linear1(combined)))
        beta = F.softmax(updated_combined, dim=1)
        sentence_output = sentence_output*beta

        dep_type_embed = self.type_embed(dep_tags_batch)
        pos_type_embed = self.pos_embed(dep_pos_batch)

        combined_embed = torch.stack((dep_type_embed,pos_type_embed),dim=0)
        pooled_embed = torch.max(combined_embed,dim=0)[0]       
        attn_1 = torch.sigmoid(self.mlp(pooled_embed))

        sentence_output = sentence_output*attn_1
        
        x = torch.tanh(self.mlp2(sentence_output))        
        x = torch.mean(x, dim=1)
        x = torch.concat([x,pool_out],dim=-1)
        logit = self.dense(x)

        return logit

class pure_bert(nn.Module):
    def __init__(self, args):
        super(pure_bert, self).__init__()
        self.args = args
        self.tokenizer=self.args.tokenizer

        self.bert = BertModel.from_pretrained(args.bert_model)
        
        
        config=self.bert.config
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, args.num_classes)
        
    def forward(self, input_cat_ids_batch, segment_ids_batch, dep_tags_batch, dep_pos_batch, dep_dist_batch, mt_idx, sent_idx):
        outputs = self.bert(input_cat_ids_batch, token_type_ids = segment_ids_batch)
        feature_output = outputs[0] # (B, S+A, D)
        pool_out = outputs[1] #(1, S+A, D)
        logit = self.dense(pool_out)

        return logit

