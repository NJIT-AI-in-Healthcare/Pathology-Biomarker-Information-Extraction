import pandas as pd
import numpy as np
import ast
import json
import os
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def get_output(input_path,output_path):
	with open(input_path) as f:
		pred_json=json.load(f)  
	y_pred=[]
	label_map={0:'O',1:'B',2:'I'}
	for ix, logit in enumerate(pred_json["logits"]):
		pred=[0]*min(99,len(pred_json["raw_X"][ix]))
		temp=pred_json["idx_map"][ix]
		if len(temp)>99:
			temp=pred_json["idx_map"][ix][:99]
		
		for jx, idx in enumerate(temp):
			# print(jx)
			lb=np.argmax(logit[jx])
			if lb==1: #B
				pred[idx]=1
			elif lb==2: #I
				if pred[idx]==0: #only when O->I (I->I and B->I ignored)
					pred[idx]=2
		y_pred.append(pred)

	# print(y_pred)

	pred_out=[]
	for i,seq in enumerate(y_pred):
		temp=[]
		for j in seq:
			temp.append(label_map[j])
		pred_out.append(temp)
	tag=pred_json["tag"]
	tag_out=[x[:99] if len(x)>99 else x for x in tag]
	# print(len(pred_out))
	# print(len(tag_out))
	df=pd.DataFrame({'y_true':tag_out,'y_pred':pred_out})
	df.to_csv(output_path)


def eval(result_path):
	result=pd.read_csv(result_path)
	# # print(result['y_true'].values.tolist())
	y_true=[ast.literal_eval(x) for x in result['y_true'].values.tolist()]
	y_pred=[[j if j!='[SEP]' else 'O' for j in ast.literal_eval(x)] for x in result['y_pred'].values.tolist()]
	report = classification_report(y_true, y_pred,digits=4,mode='strict', scheme=IOB2)
	print(report)


if __name__ == '__main__':
	result_folder = 'saved_models/BE/' # specify the folder containing the prediction
	get_output(result_folder + 'predictions.json',result_folder + 'result.csv')
	eval(result_folder + 'result.csv')


