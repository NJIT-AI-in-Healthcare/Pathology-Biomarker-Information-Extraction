import re
from nltk.tokenize import sent_tokenize
from collections import Iterable
import xmiParser as xp

###function###
def flatlist(input_list):       
    def flatten(lis):
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in flatten(item):
                    yield x
            else:        
                yield item
    output_list = list(flatten(input_list))
    return output_list

def split_sentences(x,max_len,split_mark):
    len_x = len(x.split())
    n = 1
    while len_x//n >max_len:
        n += 1
    sub_len = len_x//n
    x_l = x.split(split_mark)
    out = []
    cur = ''
    i = 0
    while i<len(x_l):
        temp = cur + x_l[i] + ' '
        if len(temp.split())>sub_len:
            if len(cur.strip())!=0:
                out.append(cur.strip())
            cur = x_l[i] + ' '
            i += 1
        else:
            cur = temp
            i+=1
    if len(cur.strip())!=0:
        out.append(cur.strip())

    return out





###main###
def text_process(text):
    section_list = text.split('\n\n')
    section_list = [x.replace('\t',' ') for x in section_list]
    section_list = [x.replace('\n','NNNNN') for x in section_list]
    section_list = [x.replace('\x18',' ') for x in section_list]
    section_list = [x.replace('\x02',' ') for x in section_list]
    section_list = [x.replace('\x03',' ') for x in section_list]
    section_list = [x.replace('"','') for x in section_list]
    section_list = [x.replace('***',' ') for x in section_list]
    section_list = [x.strip('-') for x in section_list]
    section_list = [x.strip() for x in section_list]
    section_list = [i for i in section_list if i]
    section_list = [re.sub(' +', ' ', x) for x in section_list]

    sentence_list = []
    for i in section_list:
        sentences = sent_tokenize(i)
        for j in sentences:
            sent_remove_n=j.replace('NNNNN','.')
            # if the length of sentence>100, split by '\n'; if it's still >100, split it by ',';if it's still >100, split it by ';'
            if len(sent_remove_n.split())>100:
                split_sent = split_sentences(j,100,'NNNNN')
                temp = []
                for x in split_sent:
                    if len(x.split())>100:
                        sent_split_by_comma = split_sentences(x,100,',')
                        # sent_split_by_semicolon=x.split(';')
                        for m in sent_split_by_comma:
                            if len(m.split())>100:
                                # print('here,',m)
                                sent_split_by_semicolon=split_sentences(m,100,';')
                                for p in sent_split_by_semicolon:
                                    if len(p.split())>100:
                                        print('here,',p)
                                    if p.split()!=[]:
                                        temp.append(p)
                            else:
                                if m.split()!=[]:
                                    temp.append(m)
                    else:
                        if x.split()!=[]:
                            temp.append(x)
                sentence_list.extend(temp)
            else:
                if sent_remove_n.split()!=[]:
                    sentence_list.append(sent_remove_n)

    sentence_list = flatlist(sentence_list)
    sentence_list = [x.strip('-') for x in sentence_list]
    sentence_list = [x.strip() for x in sentence_list]
    
    sen_list = []
    for sen in sentence_list:
        ori=sen
        sen = re.sub(r'(CK)(\s)(\w)', r'\1\3', sen)
        sen = re.sub(r'(CK\d)/(\d)', r'\1 CK\2', sen)
        sen = re.sub(r'(CK\d).(\d)', r'\1 CK\2', sen)
        if len(sen.split())!=0:
            sen_list.append(sen)
    
    return sen_list

def sections_sents(txtpath, report_name):
    '''
    LAST UPDATE: 07/20/2021
    '''
    txt = '.txt'
    # txtpath = '/Users/seangao/Desktop/Research/MAJOR/Rutgers/reportsfolder/txtreports/'
    # txtpath='/Users/gwt/Documents/njit/project/medical/de_ided/'
    
    with open(txtpath + report_name + txt, 'r') as file:
        report = file.read()
        
    top = report[:report.find('Clinical History')]
    top_list = text_process(top)
    top_list = [[[1, 0, 0, 0, 0],x] for x in top_list]
    
    cl = report[report.find('Clinical History'):report.find('Gross Description')]
    cl = cl.replace('Clinical History', '')
    cl_list = text_process(cl)
    cl_list = [[[0, 1, 0, 0, 0],x] for x in cl_list]
    
    gd = report[report.find('Gross Description'):report.find('Final Pathologic Diagnosis')]
    gd = gd.replace('Gross Description', '')
    gd_list = text_process(gd)
    gd_list = [[[0, 0, 1, 0, 0],x] for x in gd_list]
    
    if 'Addendum Diagnosis' in report:
        fpd = report[report.find('Final Pathologic Diagnosis'):report.find('Addendum Diagnosis')]
        fpd = fpd.replace('Final Pathologic Diagnosis', '')
        fpd_list = text_process(fpd)
        fpd_list = [[[0, 0, 0, 1, 0],x] for x in fpd_list]
        
        ad = report[report.find('Addendum Diagnosis'):]
        ad = ad.replace('Addendum Diagnosis', '')
        ad_list = text_process(ad)
        ad_list = [[[0, 0, 0, 0, 1],x] for x in ad_list]
        
        processed_list = top_list + cl_list + gd_list + fpd_list + ad_list 
        
    else:
        fpd = report[report.find('Final Pathologic Diagnosis'):]
        fpd = fpd.replace('Final Pathologic Diagnosis', '')
        fpd_list = text_process(fpd)
        fpd_list = [[[0, 0, 0, 1, 0],x] for x in fpd_list]
        
        processed_list = top_list + cl_list + gd_list + fpd_list
    return processed_list

def report2labeledlist(path, report_name):
    xmi = '.xmi'
    # xmipath = '/Users/seangao/Desktop/Research/MAJOR/Rutgers/reportsfolder/xmireportsALL0605/'
    # xmipath='/Users/gwt/Documents/njit/project/medical/annoated_all/'

    annotate = xp.xmiParser(path + 'annoated_all/' + report_name + xmi)   
    # annotate = xp.xmiParser(xmipath + report_name + xmi)

    # print(path + 'annoated_all/' + report_name + xmi)
    # print(path+'de_ided/')
    # print(annotate)
    
    if isinstance(annotate, str):
        annotate = [annotate]        
    
    units = sections_sents(path+'de_ided/', report_name)
    
    if annotate is not None:
        labeled = []
        for i in units:
            j_list = []
            for j in annotate:
                if bool(re.search(r"\b" + re.escape(j) + r"\b", i[1].upper(),re.IGNORECASE)):
                    label = 1
                    j_list.append(label)
                else:
                    label = 0
                    j_list.append(label)
            labeled.append(j_list)
    
        units_label = []
        for i in labeled:
            if sum(i) != 0:
                label = 1
                units_label.append(label)
            else: 
                label = 0
                units_label.append(label)
        
        labeled_list = list(zip(units, units_label))
        return labeled_list
    else:
        units_label = ['0'] * len(units)
        labeled_list = list(zip(units, units_label))
        return labeled_list  