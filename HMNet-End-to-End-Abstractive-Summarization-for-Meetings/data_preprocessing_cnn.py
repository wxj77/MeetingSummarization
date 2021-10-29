# Example how to transform data into the required form for simplified hmnnet
# This example processes the qmsumdata into the correct format

import json
import os
import torch

input_path = "/home/ubuntu/secondary_drive/MeetingSummarization/HMNet-End-to-End-Abstractive-Summarization-for-Meetings/cnn_raw/cnn/stories"
output_path = "/home/ubuntu/secondary_drive/MeetingSummarization/HMNet-End-to-End-Abstractive-Summarization-for-Meetings/data/train_cnn_corpus"
dict_dataset ={}
counter_datasets = 0
overall_counter = 0
dict_dataset = {}
for file in os.listdir(input_path):
    counter_datasets += 1
    if counter_datasets > 100:
        break
    fields = file.split(".story")
    input_file = open(input_path + "/" + file, "r")
    id = fields[0]
    
    
    found_summary = False
    dict_dataset[id] = {}
    dict_dataset[id]["texts"] = []
    overall_counter += 1
    utterance_counter = 0
    has_reached_summary = False
    answer = ""
    for line in input_file:
        line = line.strip()
        if line == "@highlight":
            has_reached_summary = True
        if line == "":
            utterance_counter += 1
        else:
            if has_reached_summary == False:
                utterance_list = []
                utterance_list.append(utterance_counter)
                utterance_list.append("speaker")
                utterance_list.append(line)
                dict_dataset[id]["texts"].append(utterance_list)
            else:
                if line != "@highlight":
                    answer += line + "."
    
    dict_dataset[id]["labels"] = answer

counter_dataset_entries = 0
dict_dataset_clean = {}
for key in dict_dataset:
    if len(dict_dataset[key]["texts"]) > 5 and len(dict_dataset[key]["labels"]) > 5:
        dict_dataset_clean[key] = dict_dataset[key]
for key in dict_dataset_clean:
    counter_dataset_entries += 1
    length_utterances = len(dict_dataset_clean[key]["texts"])
    #print(length_utterances)
   

print("dataset has {} entries".format(counter_dataset_entries))
torch.save(dict_dataset_clean, output_path)
    
    