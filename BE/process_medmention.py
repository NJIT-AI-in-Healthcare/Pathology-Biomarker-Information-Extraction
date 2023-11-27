import pandas as pd
import numpy as np
import ast
import json
import os
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import random
import re

# import glob
from tqdm import tqdm
import pickle
import math

import matplotlib.pyplot as plt
from collections import Counter

from nltk.tokenize import sent_tokenize
from collections import defaultdict

puncs = ',:.-()/=;+'

def punctuation(s):
	s=s.replace(',',' , ').replace(':',' : ').replace('.',' . ').replace('-',' - ').replace('(',' ( ').replace(')',' ) ').replace('/',' / ').replace('=',' = ').replace(';',' ; ').replace('+',' + ')
	return s

# extract and format documents and corresponding medical terms
# medmention dataset
f = open("corpus_pubtator.txt", "r")
save_mt = defaultdict(list)
docs = []
doc_ids = []
for i,x in enumerate(f):
  if x=='\n':
  	continue
  if '|' in x:
  	content = x.split('|')
  	if content[1] == 't':
  		doc = content[2]
  		doc_id = content[0]
  	elif content[1] == 'a':
  		doc = doc + ' ' + content[2]
  		doc = doc.replace('\n','')
  		docs.append(doc)
  		doc_ids.append(doc_id)
  else:
  	content = x.split('\t')
  	save_mt[content[0]].append((content[1], content[2], content[3]))

# print('len doc',len(docs))

# format to BIO
out = {}
global_i = 0
for i,doc_id in enumerate(doc_ids):
	mts = save_mt[doc_id]
	doc = docs[i]
	sentences = sent_tokenize(doc)
	start = 0
	j = 0 # index of mts
	for s_id, sent in enumerate(sentences):
		end = start + len(sent)
		tag = ['O']*len(sent.split())
		while j<len(mts) and int(mts[j][0])>=start and int(mts[j][1])<=end:
			char_start = int(mts[j][0]) - start
			char_end = int(mts[j][1]) - end
			tok_start = len(sent[:char_start].split())
			tok_end = len(sent[:char_end].split())
			
			if sent.split()[tok_start:tok_end] == []:
				tok_start -=1
			tag[tok_start] = 'B'
			for p in range(tok_start+1,tok_end):
				tag[p] = 'I'
			j += 1
		start = end+1
		
		new_tag = []
		new_sent = []

		for tok_i,tok in enumerate(sent.split()):
			new_tok = punctuation(tok)
			# print('new_tok',new_tok)
			if new_tok != tok:
				new_tok_l = new_tok.strip().split()
				if tag[tok_i] in ['B', 'I']:
					if len(new_tok_l) == 2:
						new_tag.extend([tag[tok_i], 'O'])
						new_sent.extend(new_tok_l)
					else:
						if tag[tok_i] == 'B':
							new_tag.extend(['B'] + ['I'] * (len(new_tok_l)-1))
							
						else:
							new_tag.extend(['I'] * (len(new_tok_l)))
						if new_tok_l[-1] in puncs:
								new_tag[-1] = 'O'
						new_sent.extend(new_tok_l)
				else:
					new_tag.extend(['O']*len(new_tok_l))
					new_sent.extend(new_tok_l)
			else:
				new_tag.append(tag[tok_i])
				new_sent.append(sent.split()[tok_i])
		out[str(global_i)] = {'label': new_tag, 'sentence': new_sent}
		global_i += 1
json.dump(out, open('medmention.json', 'w' ) )



