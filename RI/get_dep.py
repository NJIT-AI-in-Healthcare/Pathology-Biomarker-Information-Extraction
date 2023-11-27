import argparse
import json
import os
import re
import sys
from allennlp.predictors.predictor import Predictor
from lxml import etree
from tqdm import tqdm
import pandas as pd
import ast
import spacy
import re


def get_dependencies_split_mt(data, predictor,nlp):
    data = text2docs_split_mt(data, predictor,nlp)
    data=dependencies2format_split_mt_new(data,nlp)
    return data

def text2docs_split_mt(data, predictor,nlp):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    data['dependency']=None
    print('Predicting dependency information...')
    for index,row in tqdm(data.iterrows()):
        content=row['unit']
        sentences= nlp(content)
        out=predictor.predict(sentence=sentences)
        data.at[index,'dependency']=out
    return data

def dependencies2format_split_mt_new(data,nlp):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    for index,row in tqdm(data.iterrows()):
        doc=row['dependency']
        mt=row['biomarker'].lower()
        note=row['unit'].lower().split()
        sentence = {}
        sentence['tokens'] = doc['words']
        sentence['tags'] = doc['pos']
        # sentence['energy'] = doc['energy']
        predicted_dependencies = doc['predicted_dependencies']
        predicted_heads = doc['predicted_heads']
        sentence['predicted_dependencies'] = doc['predicted_dependencies']
        sentence['predicted_heads'] = doc['predicted_heads']
        sentence['dependencies'] = []
        sentence['aspect']=[]
        sent_tokens=nlp(' '.join(sentence['tokens']))
        for idx, item in enumerate(predicted_dependencies):
            dep_tag = item
            frm = predicted_heads[idx]
            to = idx + 1
            sentence['dependencies'].append([dep_tag, frm, to])
        nouns=[]
        found_flag=0
    
        flag=0
        tokenize_mt=[w.text for w in nlp(mt)]
        res = re.search(re.escape(' '.join(tokenize_mt)),sent_tokens.text)
        checked_token = [re.sub(r'[^\w\s]','',x) for x in sentence['tokens']]
        if res:
            left=len(sent_tokens.text[:res.start()].split())
            right=len(sent_tokens.text[:res.end()].split())
            nouns.append([left,right])
            sentence['aspect'].append(mt)
            found_flag=1
        else:
        	print('no results,',re.escape(' '.join(tokenize_mt)),' '.join(tokenize_mt),sent_tokens.text)
        sentence['from_to']=nouns
        data.at[index,'dependency']=sentence
    return data


def main():
    data_path = 'biomarker_data/'
    model_path='https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz'
    output_path=data_path+'dependency/'

    predictor = Predictor.from_path(model_path)
    data = [('rule_based_data_train.csv', 'rule_based_data_test.csv')]
    for train_file, test_file in data:
        train=pd.read_csv(data_path+train_file)
        test=pd.read_csv(data_path+test_file)

        nlp = spacy.load("en_core_web_sm")
        train = get_dependencies_split_mt(train, predictor,nlp)
        test = get_dependencies_split_mt(test, predictor,nlp)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_folder=output_path
        print('output_folder',output_folder)


        train.to_csv(output_folder+train_file)
        test.to_csv(output_folder+test_file)


if __name__ == "__main__":
    main()