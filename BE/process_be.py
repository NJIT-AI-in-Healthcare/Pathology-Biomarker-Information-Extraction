import pandas as pd
import numpy as np
import ast
import json
import os
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import random
import xmiParser as xp
import re

import glob
from tqdm import tqdm
import data_until as r2ll
import pickle
import math

import matplotlib.pyplot as plt
from collections import Counter

def punctuation(s):
	s=s.replace(',',' , ').replace(':',' : ').replace('.',' . ').replace('-',' - ').replace('(',' ( ').replace(')',' ) ').replace('/',' / ').replace('=',' = ').replace(';',' ; ').replace('+',' + ')
	return s

def norm_gt(gt,text):
	new_gt=[]
	for b in gt:
		matched_bio=[m.start() for m in re.finditer(b,text)]
		for i in range(len(matched_bio)):
			new_gt.append(b)
	return new_gt

def generate_folds(input_folder, output_folder):
	# xmipath = glob.glob('/Users/gwt/Library/CloudStorage/OneDrive-NJIT/My_laptop/NJIT/project/biomarker/data/annoated_all/*.xmi')
	xmipath = glob.glob(input_folder + 'annoated_all/' + '/*.xmi')
	xminame = []
	for i in xmipath:
	    n = i[i.rfind('/')+1:]
	    n = n[:n.rfind('.')]
	    xminame.append(n)

	r2llList = []
	for i in tqdm(xminame):
	    r2llres = r2ll.report2labeledlist(input_folder, i)
	    r2llList.append([i, r2llres])

	# n = 209
	n = math.ceil(len(xminame)/5)
	# print('n', n)
	all_folds = [xminame[i:i + n] for i in range(0, len(xminame), n)]

	fold = []
	for i in xminame:
	    if i in all_folds[0]:
	        f = 0
	        fold.append(f)
	    elif i in all_folds[1]:
	        f = 1
	        fold.append(f)
	    elif i in all_folds[2]:
	        f = 2
	        fold.append(f)
	    elif i in all_folds[3]:
	        f = 3
	        fold.append(f)
	    else:
	        f = 4
	        fold.append(f)
        
	with open(output_folder+'data.pkl', 'wb') as f:
	    pickle.dump(r2llList, f)        
	        
	        
	with open(output_folder+'data_fold.pkl', 'wb') as f:
	    pickle.dump(fold, f)         

def format_data(raw_folder, folder):
	# data=pd.read_pickle('0803/data0803.pkl')
	# fold=pd.read_pickle('0803/data_fold0803.pkl')
	data=pd.read_pickle(folder+'/data.pkl')
	fold=pd.read_pickle(folder+'/data_fold.pkl')
	report_re=[]
	sentence_re=[]
	section_re=[]
	biomarker_re=[]
	fold_re=[]
	print(len(data))
	# report_wrong=[]
	for idx,report in enumerate(data):
		all_bios=xp.xmiParser(raw_folder + report[0]+'.xmi')
		if type(all_bios)==type(''):
			all_bios=[all_bios]
		sentences=report[1]
		b=0
		for sent_info in sentences:

			section=sent_info[0][0]
			sent=sent_info[0][1].lower()
			# print(section)
			# print(sent)
			# sent_list=sent.split()
			bio_list=[]
			# print(sent_info[1])
			if sent_info[1]==1:
				for bio in all_bios:
					bio_low=bio.lower()
					if bio_low in sent:
						if bio_low=='er' or bio_low=='pr' or bio_low=='igh':
							if bio_low in punctuation(sent).split():
								bio_list.append(bio_low)
						else:
							bio_list.append(bio_low)
						
				# if len(bio_list)==0:
				# 	if report[0] not in report_wrong:
				# 		report_wrong.append(report[0])
				if len(bio_list)==0:
					print(all_bios)
					print(sent)
					print(sent_info)
			if len(bio_list)>1:
				bio_list=sorted(list(set(bio_list)),key=len)

				for i,b in enumerate(bio_list):
					if i!=len(bio_list)-1:
						for p in bio_list[i+1:]:
							if b in p:
								if sent.count(b)==1:
									bio_list[i]='****'
				while '****' in bio_list:
					bio_list.remove('****')
			if (len(bio_list) == 0 and len(sent.split())<3) or sent == '[name] - [idnum]':
				continue

			report_re.append(report[0])
			sentence_re.append(sent)
			section_re.append(section)
			biomarker_re.append(bio_list)
			fold_re.append(fold[idx])
	df=pd.DataFrame({'report_name':report_re,'sentence':sentence_re,'section':section_re,'biomarker':biomarker_re,'fold':fold_re})
	df.to_csv(folder+'/data.csv')
	return df

def generate_data_for_bert_util(input_folder):
	data=pd.read_csv(input_folder+'/data.csv')
	sent_re=[]
	fold_re=[]
	# section_re=[]
	for index,row in data.iterrows():
		
		bio_list=ast.literal_eval(row['biomarker'])

		sent=row['sentence']
		bio_list=norm_gt(bio_list,sent)
		if pd.isna(sent):
			continue
		else:
			sent=punctuation(sent).split()
		section=ast.literal_eval(row['section'])
		fold=row['fold']

		tag=['O']*len(sent)

		if len(bio_list)!=0:
			temp=[]
			for b in bio_list:
				temp.append(punctuation(b).split())
			temp_first=[x[0] for x in temp]

			i=0
			count=0
			while i<len(sent):
				if sent[i] in temp_first:
					ori_i=i
					for bio in temp: #find the matched biomarker
						if sent[i]==bio[0]:
							if len(bio)==1:
								tag[i]='B'
								i+=1
								count+=1
								break
							
							elif len(bio)>1:
								if sent[i:i+len(bio)]==bio:
									tag[i]='B'
									p=1
									while p<=len(bio)-1:
										tag[p+i]='I'
										p+=1
									if (p+i)!=i+len(bio):
										print('+')
									i=i+len(bio)
									count+=1

									break
					if i==ori_i:
						i+=1
				else:
					i+=1

		sent_re.append((sent,tag))
		fold_re.append(fold)
		# section_re.append(section)


	# re_df=pd.DataFrame({'sent':sent_re,'fold':fold_re,'section':section_re})
	re_df=pd.DataFrame({'sent':sent_re,'fold':fold_re})
	# print(re_df)
	re_df.to_csv(output_folder+'/all.csv')
	return re_df

def generate_data_for_bert(data,output_path,fold_num):
	# data=pd.read_csv('output_0713/new_bio_10fold.csv')
	# data=pd.read_csv(data_path)
	for fold in range(fold_num):
	# fold=1
		train_and_val={}
		test={}
		i=0
		j=0
		idx_record=[]
		for index,row in data.iterrows():
			# re=ast.literal_eval(row['sent'])
			re=row['sent']
			if row['fold']!=fold:
				train_and_val[str(i)]={"label":re[1],"sentence": re[0]}
				idx_record.append(str(i))
				i+=1
			else:
				test[str(j)]={"label":re[1],"sentence": re[0]}
				j+=1
			

		res = random.sample(range(len(train_and_val)), int(len(train_and_val)*0.1))


		train={}
		val={}
		p=0
		q=0
		for j in range(len(train_and_val)):
			key=idx_record[j]
			value=train_and_val[key]
			if j in res:
				val[str(p)]=value
				p+=1
			else:
				train[str(q)]=value
				q+=1
		if not os.path.exists(output_path+'fold_'+str(fold)):
			os.makedirs(output_path+'fold_'+str(fold))

		json.dump(train, open(output_path+'fold_'+str(fold)+'/train.json', 'w' ) )
		json.dump(test, open(output_path+'fold_'+str(fold)+'/test.json', 'w' ) )
		json.dump(val, open(output_path+'fold_'+str(fold)+'/dev.json', 'w' ) )

def get_statistics_ae(data):
	len_total = len(data)
	data['bio_len'] = data['biomarker'].map(len)
	print(data.head())
	sub_data = data[data['bio_len']>0]
	data['sent_len'] = data['sentence'].apply(lambda x: len(x.split()))
	print('sent len',data['sent_len'].describe())
	# sub_data = data[data['biomarker'].map(len)==0]
	# print(type(data.iloc[0]['biomarker']))
	# print(sub_data['biomarker'])
	print('Total number of sentences,', len_total)
	print('Number of sentences with biomarkers,', len(sub_data))
	print('Rato of the sentences containing biomarkers,', len(sub_data)/len_total)
	# print(data['bio_len'].describe())
	# data['bio_len'].hist()
	# plt.show()
	statistics = dict(Counter(data['bio_len'].values.tolist()))
	print(dict(sorted(statistics.items())))
	print('Number of sentences with more than 3 biomarkers,',sum([v for k,v in statistics.items() if k>=4]))
	print('The ratio of the sentences containing more than 3 biomarkers', sum([v for k,v in statistics.items() if k>=4])/sum([v for k,v in statistics.items() if k!=0]))







if __name__ == '__main__':
	input_folder = '../../data/' # the folder containing the annotated reports (.xmi)
	output_folder = 'data_process/' # the folder to save the preprocessed data
	#step 1: split data to folds
	# generate_folds(input_folder, output_folder)

	#step 2: format data
	# data = format_data(input_folder + '/annoated_all/', output_folder)
	# get_statistics_ae(data)

	#step 3: Convert data to BIO format
	# df = generate_data_for_bert_util(output_folder)
	# generate_data_for_bert(df, output_folder, 5)




