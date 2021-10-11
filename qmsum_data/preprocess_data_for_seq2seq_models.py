import json
import nltk
from nltk import word_tokenize

# filter some noises caused by speech recognition
def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text
    
def preprocess_data_for_bart_all_meeting_content(data, split):
    bart_data = []
    for i in range(len(data)):
    # get meeting content
        src = []
        for k in range(len(data[i]['meeting_transcripts'])):
            cur_turn = data[i]['meeting_transcripts'][k]['speaker'].lower() + ': '
            cur_turn = cur_turn + tokenize(data[i]['meeting_transcripts'][k]['content'])
            src.append(cur_turn)
        src = ' '.join(src)
        for j in range(len(data[i]['general_query_list'])):
            cur = {}
            query = tokenize(data[i]['general_query_list'][j]['query'])
            cur['src'] = clean_data('<s> ' + query + ' </s> ' + src + ' </s>')
            target = tokenize(data[i]['general_query_list'][j]['answer'])
            cur['tgt'] = target
            bart_data.append(cur)
        for j in range(len(data[i]['specific_query_list'])):
            cur = {}
            query = tokenize(data[i]['specific_query_list'][j]['query'])
            cur['src'] = clean_data('<s> ' + query + ' </s> ' + src + ' </s>')
            target = tokenize(data[i]['specific_query_list'][j]['answer'])
            cur['tgt'] = target
            bart_data.append(cur)
        
    print('Total {} query-summary pairs in the {} set'.format(len(bart_data), split))
    print(bart_data[2])
    with open('data/bart_' + split + '.jsonl', 'w') as f:
        for i in range(len(bart_data)):
            print(json.dumps(bart_data[i]), file=f)
            

def preprocess_data_for_bart_gold_spans(data, split):
    # process data for BART
    # the input of the model here is the gold span corresponding to each query
    bart_data_gold = []
    for i in range(len(data)):
    # get meeting content
        entire_src = []
        for k in range(len(data[i]['meeting_transcripts'])):
            cur_turn = data[i]['meeting_transcripts'][k]['speaker'].lower() + ': '
            cur_turn = cur_turn + tokenize(data[i]['meeting_transcripts'][k]['content'])
            entire_src.append(cur_turn)
        entire_src = ' '.join(entire_src)
        for j in range(len(data[i]['general_query_list'])):
            cur = {}
            query = tokenize(data[i]['general_query_list'][j]['query'])
            cur['src'] = clean_data('<s> ' + query + ' </s> ' + entire_src + ' </s>')
            target = tokenize(data[i]['general_query_list'][j]['answer'])
            cur['tgt'] = target
            bart_data_gold.append(cur)
        for j in range(len(data[i]['specific_query_list'])):
            cur = {}
            query = tokenize(data[i]['specific_query_list'][j]['query'])
            src = []
            # get the content in the gold span for each query
            for span in data[i]['specific_query_list'][j]['relevant_text_span']:
                assert len(span) == 2
                st, ed = int(span[0]), int(span[1])
                for k in range(st, ed + 1):
                    cur_turn = data[i]['meeting_transcripts'][k]['speaker'].lower() + ': '
                    cur_turn = cur_turn + tokenize(data[i]['meeting_transcripts'][k]['content'])
                    src.append(cur_turn)
            src = ' '.join(src)
            cur['src'] = clean_data('<s> ' + query + ' </s> ' + src + ' </s>')
            target = tokenize(data[i]['specific_query_list'][j]['answer'])
            cur['tgt'] = target
            bart_data_gold.append(cur)
        
    print('Total {} query-summary pairs in the {} set'.format(len(bart_data_gold), split))
    print(bart_data_gold[2])
    with open('data/bart_' + split + '._gold.jsonl', 'w') as f:
        for i in range(len(bart_data_gold)):
            print(json.dumps(bart_data_gold[i]), file=f)

def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens

def main():
    nltk.download('punkt')
    split = 'train'
    data_path = 'data/ALL/jsonl/' + split + '.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    n_meetings = len(data)
    print('Total {} meetings in the {} set.'.format(n_meetings, split))
    print('Preprocessing all meeting data for Bart. Output in data folder.')
    preprocess_data_for_bart_all_meeting_content(data, split)
    preprocess_data_for_bart_gold_spans(data, split)
    
if __name__ == "__main__":
    main()