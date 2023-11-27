import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# from torch.utils.data import DataLoader, random_split, ConcatDataset

def resultNorm(input_string):
    result_lookup = pd.ExcelFile('Biomarker_result_lookup.xlsx')
    result_positive = pd.read_excel(result_lookup, 'Positive', header=None)
    result_negative = pd.read_excel(result_lookup, 'Negative', header=None)
    
    positive_list = ['Positive'] * result_positive.shape[0]
    result_positive['standard_result'] = positive_list
    result_positive.columns = ['raw_result', 'standard_result']
    
    negative_list = ['Negative'] * result_negative.shape[0]
    result_negative['standard_result'] = negative_list
    result_negative.columns = ['raw_result', 'standard_result']
    
    list_positive = result_positive['raw_result'].to_list()
    list_negative = result_negative['raw_result'].to_list()

    for neg in list_negative:
        input_string = re.sub(r'\b' + neg + r'\b', 'Negative', input_string)
    for pos in list_positive:
        input_string = re.sub(r'\b' + pos + r'\b', 'Positive', input_string)        
    
    return input_string       

def checkWindow(input_biomarker, unit):
    input_biomarker = input_biomarker.upper()
    unit = unit.upper()
    unit = unit.replace('WHILE', '.')
    before_part = unit[:unit.find(input_biomarker)]
    if '.' in before_part and '5.2' not in before_part and '19.9' not in before_part and '5.6' not in before_part and '8.18' not in before_part:
        before_part_short = before_part[before_part.rfind('.'):]
    else:
        before_part_short = before_part
    
    after_part = unit[unit.find(input_biomarker):]
    if '.' in after_part and '5.2' not in after_part and '19.9' not in after_part and '5.6' not in after_part and '8.18' not in after_part:
        after_part_short = after_part[:after_part.find('.')]
    else:
        after_part_short = after_part
        
    checkwindow = before_part_short + ' ' + after_part_short
    checkwindow = checkwindow.replace(' \n', ' ')
    checkwindow = checkwindow.replace('\n', ' $$$$$$$$$$$$$$$ ')
    checkwindow = checkwindow.replace('\t', ' ')
#    checkwindow = checkwindow.replace('\n', '')
    checkwindow = re.sub(' +', ' ', checkwindow)
    checkwindow = checkwindow.replace(',', ' $$$$$$$$$$$$$$$')
    checkwindow = checkwindow.replace(';', ' $$$$$$$$$$$$$$$')
    checkwindow = checkwindow.replace('AND', '$$$$$$$$$$$$$$$')
    checkwindow = checkwindow.replace('FOR', '$')
    checkwindow = checkwindow.upper()
    return checkwindow

def commaWeight(list_bios, unit):
    unitList = unit.split(' ')
    indices = [i for i, x in enumerate(unitList) if x == '$$$$$$$$$$$$$$$']
    for i in indices:
        if unitList[i-1] in list_bios and unitList[i+1] in list_bios:
            unitList[i] = ','
    unitOut = ' '.join(unitList)
    unitOut = unitOut.replace(' , ', ', ')
    return unitOut

def findClosestResult(biomarker, unit, list_bios):
    result_window = checkWindow(biomarker, unit)
    result_window = commaWeight(list_bios, result_window)
    result_window = resultNorm(result_window)
    position_biomarker = result_window.find(biomarker.upper())    
    position_positive = [m.start() for m in re.finditer('Positive', result_window)]
    position_negative = [m.start() for m in re.finditer('Negative', result_window)]
    
    if 'Positive' in result_window or 'Negative' in result_window:
        if len(position_positive)!=0:
            positive_dist = []
            for i in position_positive:
                dist = abs(i-position_biomarker)
                positive_dist.append(dist)
        else:
            positive_dist = [9999]
        
        if len(position_negative)!=0:
            negative_dist = []
            for i in position_negative:
                dist = abs(i-position_biomarker)
                negative_dist.append(dist)
        else:
            negative_dist = [9999] 
            
        if min(positive_dist) > min(negative_dist):
            result = 'Negative'
        else:
            result = 'Positive'
    else:
        result = 'No Result'
        
    return result

def run_rule_based(test,number):
    data=pd.read_csv('../biomarker_data/0608/rule_based_data.csv')
    y_true=[]
    y_predict=[]
    # print(data_set)
    # print(len(data_set))

    wrong_bio=[]
    wrong_unit=[]
    wrong_true=[]
    wrong_predict=[]
    test_id=[]
    for batch in test:
        # print(batch['s_id'])
        for i in batch['s_id']:
            # print(i)
            test_id.append(int(i))
    print('test_set_len:',len(test_id))
    input_data=data[data['sent_id'].isin(test_id)]
        # for i in t_batch:
        #     print(count)
        #     print(i)
        #     count+=1
    for index,row in input_data.iterrows():
        a=findClosestResult(row['biomarker'], row['unit'], row['bio_list'])
        if a=='Positive':
            y_predict.append(1)
            predict=1
        elif a=='Negative':
            y_predict.append(-1)
            predict=-1
        else:
            y_predict.append(0)
            predict=0
        y_true.append(row['polarity'])
        if row['polarity']!=predict:
            wrong_bio.append(row['biomarker'])
            wrong_unit.append(row['unit'])
            wrong_true.append(row['polarity'])
            wrong_predict.append(predict)

    result=precision_recall_fscore_support(y_true, y_predict, average='macro')
    accuracy=accuracy_score(y_true, y_predict)
    id_df=pd.DataFrame({'sent_id':test_id})
    id_df.to_csv('0608/10fold/fold_'+str(number)+'_sent_id.csv')
    # df=pd.DataFrame({'biomarker':wrong_bio,'unit':wrong_unit,'true':wrong_true,'predict':wrong_predict})
    # df.to_csv('rule_result.csv')
    # print(y_predict)
    # print(y_true)
    print('fold'+str(number),result)
    print(accuracy)

def record_train_test_id(test,number):
    test_id=[]
    for batch in test:
        # print(batch)
        # print(batch[-1])
        # print(batch[-1][0])
        for i in batch[-1]:
        # for i in batch['s_id']:
            print(i)
            test_id.append(int(i))
    print('test_set_len:',len(test_id))
    id_df=pd.DataFrame({'sent_id':test_id})
    id_df.to_csv('0620/fold_'+str(number)+'_sent_id.csv')

if __name__ == '__main__':
    input_data=pd.read_csv('biomarker_data/0529/rule_based_data_test.csv')
    y_true=[]
    y_predict=[]

    wrong_bio=[]
    wrong_unit=[]
    wrong_true=[]
    wrong_predict=[]
    for index,row in input_data.iterrows():
        a=findClosestResult(row['biomarker'], row['unit'], row['bio_list'])
        if a=='Positive':
            y_predict.append(1)
            predict=1
        elif a=='Negative':
            y_predict.append(-1)
            predict=-1
        else:
            y_predict.append(0)
            predict=0
        y_true.append(row['polarity'])
        if row['polarity']!=predict:
            wrong_bio.append(row['biomarker'])
            wrong_unit.append(row['unit'])
            wrong_true.append(row['polarity'])
            wrong_predict.append(predict)

    result=precision_recall_fscore_support(y_true, y_predict, average='macro')
    accuracy=accuracy_score(y_true, y_predict)
    df=pd.DataFrame({'biomarker':wrong_bio,'unit':wrong_unit,'true':wrong_true,'predict':wrong_predict})
    df.to_csv('rule_result.csv')
    # print(y_predict)
    # print(y_true)
    print(result)
    print(accuracy)



'''
testBio = 'Synaptophysin'
testUnit = 'NOTE.	The tumor cells are positive for CK8.18, Synaptophysin and TTF-1.'
testList = ['GATA3', 'Uroplakin II', 'CK7', 'CD138', 'GCDFP/mammaglobin']

findClosestResult(testBio, testUnit, testList)
'''


