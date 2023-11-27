import xmltodict
from collections import Iterable

###functions###
def tabfix(input_str):
    o = input_str.replace('\t', ' ')
    return o

def her2fix(input_string):
    if 'HER 2' in input_string or 'HER-2' in input_string:
        o = 'HER2'
        return o
    else:
        return input_string
    
def ckfix(input_str):
    if input_str.startswith('CK') and '.' in input_str:
        ck1 = input_str[:input_str.find('.')]
        ck2 = 'CK' + input_str[input_str.find('.')+1:]
        return [ck1.replace(' ',''), ck2.replace(' ','')]
    elif input_str.startswith('CK') and '/' in input_str:
        ck1 = input_str[:input_str.find('/')]
        ck2 = 'CK' + input_str[input_str.find('/')+1:]
        return [ck1.replace(' ',''), ck2.replace(' ','')]
    elif input_str.startswith('CK') and '&' in input_str:
        ck1 = input_str[:input_str.find('&')]
        ck2 = 'CK' + input_str[input_str.find('&')+1:]
        return [ck1.replace(' ',''), ck2.replace(' ','')]
    elif input_str.startswith('CK') and ';' in input_str:
        ck1 = input_str[:input_str.find(';')]
        ck2 = 'CK' + input_str[input_str.find(';')+1:]
        return [ck1.replace(' ',''), ck2.replace(' ','')]
    elif input_str.startswith('CYTOKERATIN') and '/' in input_str:
        ck1 = 'CK' + input_str[input_str.find(' ')+1:input_str.find('/')]
        ck2 = 'CK' + input_str[input_str.find('/')+1:]
        return [ck1.replace(' ',''), ck2.replace(' ','')]
    else:
        return input_str
    
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

def gmfix(input_str):
    if 'GC' in input_str.upper() and 'MA' in input_str.upper():
        o = 'GCDFP-15/Mammaglobin'
        return o
    else:
        return input_str
    # if 'gcdfp-15/mammoglobin' in input_str.lower() or 'gcdf/mammoglobin' in input_str.lower() or 
    # 'gcdf/mam' in input_str.lower() or 'gcdfp15/mammaglobin' in input_str.lower() or 'mammoglobin/gcdp15' in input_str.lower():
    #     o = 'GCDFP-15/Mammaglobin'
    #     return o
    # else:
    #     return input_str

###main###
def xmiParser(filepath):
    # with open('annoated_all/'+filepath) as in_file:
    with open(filepath) as in_file:
        xml = in_file.read()
        
    xml_json = xmltodict.parse(xml)
    
    if 'typesystem:ClampNameEntityUIMA' in xml_json['xmi:XMI']:
        senjson_list = xml_json['xmi:XMI']['textspan:Sentence']
        senkey_list = []
        for i in senjson_list:
            b = i['@begin']
            e = i['@end']
            sen_id = i['@sentenceNumber']
            senkey_list.append([b, e, sen_id])
            
        body = xml_json['xmi:XMI']['cas:Sofa']['@sofaString']
        body_list = []
        for i in senkey_list:
            sen_id = i[2]
            sen_i = body[int(i[0]):int(i[1])]
            body_list.append([sen_id, sen_i])
        
        checker = xml_json['xmi:XMI']['typesystem:ClampNameEntityUIMA']
        
        if type(checker) == list:
            entjson_list = xml_json['xmi:XMI']['typesystem:ClampNameEntityUIMA']
            ent_list = []
            for i in entjson_list:
                e_id = i['@xmi:id']
                e = body[int(i['@begin']):int(i['@end'])]
                e_tag = i['@semanticTag']
                ent_list.append([e_id, e, e_tag])
                
                output_list = []
                for i in ent_list:
                    if i[2] == 'biomarker':
                        o = i[1]
                        output_list.append(o)
                    else:
                        pass
            # print(output_list)
                    
            output_list = list(set(i.upper() for i in output_list))
            output_list_fix = []
            for i in output_list:
                # print(i)
                o = tabfix(i)
                o = her2fix(o)
                o = gmfix(o)
                o = ckfix(o)
                # print(o)
                output_list_fix.append(o)
            output_list_fix = flatlist(output_list_fix)
            output_list_fix = list(set(output_list_fix))
            output_list_fix = [x.strip() for x in output_list_fix]
            return output_list_fix
            
        else:
            ent = xml_json['xmi:XMI']['typesystem:ClampNameEntityUIMA']
            e_id = ent['@xmi:id']
            e = body[int(ent['@begin']):int(ent['@end'])]
            e_tag = ent['@semanticTag']
            
            e = e.upper()
            e = tabfix(e)
            e = her2fix(e)
            e = gmfix(e)
            e = ckfix(e)
            return e
    else:
        return None   