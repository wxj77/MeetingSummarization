# Example how to transform data into the required form for simplified hmnnet
# This example processes the qmsumdata into the correct format

import json
import os
import torch

input_path = "/home/ubuntu/secondary_drive/MeetingSummarization/qmsum_data/data/ALL/train"
output_path = "/home/ubuntu/secondary_drive/MeetingSummarization/HMNet-End-to-End-Abstractive-Summarization-for-Meetings/data/train_qmsum_data_queries_corpus"
dict_dataset ={}
counter_datasets = 0
overall_counter = 0
for file in os.listdir(input_path):
    counter_datasets += 1
    if counter_datasets > 300:
        break
    fields = file.split(".json")
    input_file = open(input_path + "/" + file, "r")
    file_content = input_file.read()
    file_content = json.loads(file_content)
    id_prel = fields[0]
    
    
    found_summary = False
    for obj in file_content["specific_query_list"]:
        query = obj["query"]
        id = id_prel + str(overall_counter)
        overall_counter += 1
        dict_dataset[id] = {}
        dict_dataset[id]["texts"] = []
        answer = obj["answer"]
        relevant_text_spans = obj["relevant_text_span"]
        #print(relevant_text_spans)
        relevant_text_span_begin = int(relevant_text_spans[0][0])
        relevant_text_span_end = int(relevant_text_spans[0][1])
        dict_dataset[id]["labels"] = answer
        utterance_counter = 1
        for counter, transcript in enumerate(file_content["meeting_transcripts"]):
            role = transcript["speaker"]
            utterance_counter += 1
            if utterance_counter > relevant_text_span_begin and utterance_counter < relevant_text_span_end:
                utterance = transcript["content"]
                utterance_list = []
                utterance_list.append(utterance_counter)
                utterance_list.append(role)
                utterance_list.append(utterance)
                dict_dataset[id]["texts"].append(utterance_list)
        #if utterance_counter > relevantextst_text_span_begin and utterance_counter < relevant_text_span_end:
        #        dict_dataset[id][""].append(utterance_list)
        #        print("Length utterance list {}".format(len(utterance_list)))

#print(dict_dataset)
counter_dataset_entries = 0
dict_dataset_clean = {}
for key in dict_dataset:
    if len(dict_dataset[key]["texts"]) > 5:
        dict_dataset_clean[key] = dict_dataset[key]
for key in dict_dataset_clean:
    counter_dataset_entries += 1
    length_utterances = len(dict_dataset_clean[key]["texts"])
    print(length_utterances)
   

print("dataset has {} entries".format(counter_dataset_entries))
torch.save(dict_dataset_clean, output_path)