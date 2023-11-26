# Pathology-Biomarker-Information-Extraction
## About The Project
Codes and models to extract cancer biomarkers and results from pathology reports. 



## Quick strat: Load the trained models for extraction
If you want to skip training and directly load the model to extract biomarkers and results

1. Extract biomarkers: 
```python BE/run_be.py --bert_model saved_models/BIEMP/ --do_eval --max_seq_length 128 --output_dir PATH_TO_OUTPUT --data_dir PATH_TO_DATA```

2. Extract results:
```python RI/run_ri.py --bert_model saved_models/BIEMP/ --data_dir PATH_TO_DATA --output_dir PATH_TO_OUTPUT --dropout 0.3 --learning_rate 2e-5 --num_train_epochs 2 --mode biempa --load_model --load_classification_path saved_models/BIEMPA/model/checkpoint_0/```
```python RI/run_ri.py --bert_model saved_models/BIEMP/ --data_dir data/RI/ --output_dir output/RI/ --dropout 0.3 --learning_rate 2e-5 --num_train_epochs 2 --mode biempa --load_model --load_classification_path saved_models/BIEMPA/model/checkpoint_0/```

## Whole process: data preparation, training, extraction 
If you want to go through the whole process, please follow the following steps:


### Biomarker Extraction
1. Data Preparation:
   The input data format of post-training and fine-tuning: `{sentence_index: {'label': BIO format, 'sentence': list of words}}`
   ```
   {
   '0': {'label': ['O', 'B', 'I','I','O','O'], 'sentence': ['The', 'shortness', 'of', 'breath', 'improved','.']},
   '1': {'label': ['O', 'B', 'I','I','O','O'], 'sentence': ['The', 'shortness', 'of', 'breath', 'improved','.']},
   }
   ```
   Scripts:
   - preprocess post-training data: `process_medmention.py`
   - preprocess fine-tuning data: `process_be.py`

3. Post-training using Medmention dataset
   ```
   python BE/run_pt.py --bert_model PATH_TO_BIOBERT_MODEL --max_seq_length 128 --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 10 --output_dir PATH_TO_POSTTRAINED_MODEL --data_dir PATH_TO_Biomarker_Extraction_DATA
   ```

4. Fine-tuning on pathology reports (`BE/run_be.sh`)

5. Evaluation (`BE/eval_be.py`)

### Result Identification
1. Data preparation:
   The input data format:
   ```
   - biomarker
   - unit: sentence
   - polarity: the result of biomarker
   - dependency: dependency features extracted by CoreNLP
   ```
   Script to get the dependency: `get_dep.py`

2. Identify results
   ```
   # train
   python RI/run_ri.py --bert_model PATH_TO_POSTTRAINED_MODEL --data_dir PATH_TO_Result_Identification_DATA --output_dir PATH_TO_OUTPUT_FOLDER --dropout 0.3 --learning_rate 5e-5 --num_train_epochs 2 --mode biempa

   # load model
   python RI/run_ri.py --bert_model PATH_TO_POSTTRAINED_MODEL --data_dir PATH_TO_Result_Identification_DATA --output_dir PATH_TO_OUTPUT_FOLDER --dropout 0.3 --learning_rate 5e-5 --num_train_epochs 2 --mode biempa --load_model --load_classification_path
   PATH_TO_THE_SAVED_MODEL
   ```

### Built With
* [pytorch](https://pytorch.org/)
* [transformers](https://huggingface.co/transformers/v4.7.0/installation.html)
* [pandas](https://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [seqeval](https://pypi.org/project/seqeval/0.0.10/)

