# Example how to transform data into the required form for simplified hmnnet
# This example processes the qmsumdata into the correct format

import json
import os
import torch

input_path = "/home/ubuntu/secondary_drive/MeetingSummarization/qmsum_data/data/ALL/test"
output_path = "/home/ubuntu/secondary_drive/MeetingSummarization/HMNet-End-to-End-Abstractive-Summarization-for-Meetings/data/test_qmsum_data_corpus"
dict_dataset ={}

for file in os.listdir(input_path):
    fields = file.split(".json")
    input_file = open(input_path + "/" + file, "r")
    file_content = input_file.read()
    file_content = json.loads(file_content)
    id = fields[0]
    
    found_summary = False
    for obj in file_content["general_query_list"]:
        if obj["query"] == "Summarize the whole meeting.":
            dict_dataset[id] = {}
            dict_dataset[id]["texts"] = []
            answer = obj["answer"]
            dict_dataset[id]["labels"] = answer
            found_summary = True
            #print("summary found")
    
    if found_summary == True:
        for counter, transcript in enumerate(file_content["meeting_transcripts"]):
            role = transcript["speaker"]
            utterance_counter = counter + 1
            utterance = transcript["content"]
            utterance_list = []
            utterance_list.append(utterance_counter)
            utterance_list.append(role)
            utterance_list.append(utterance)
            dict_dataset[id]["texts"].append(utterance_list)

#print(dict_dataset)
counter_dataset_entries = 0
for key in dict_dataset:
    counter_dataset_entries += 1

print("dataset has {} entries".format(counter_dataset_entries))
torch.save(dict_dataset, output_path)