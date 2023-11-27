#!/bin/bash

# Please specify the folder before running
# train
n=4
for (( i=0 ; i<=$n ; i++ )); 
do
	python run_be.py --bert_model /biobert --do_train --do_valid --max_seq_length 128 --train_batch_size 16 --learning_rate 5e-5 --num_train_epochs 1 --output_dir data_process/result/biobert/fold_$i/ --data_dir data_process/fold_$i/

	python run_be.py --bert_model /biobert --do_eval --max_seq_length 128 --output_dir data_process/result/biobert/fold_$i/ --data_dir data_process/fold_$i/
done

# # load model
# n=4
# for (( i=0 ; i<=$n ; i++ )); 
# do
# 	python run_be.py --bert_model /biobert --do_eval --max_seq_length 128 --output_dir data_process/result/biobert/fold_$i/ --data_dir data_process/fold_$i/
# done