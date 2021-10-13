# Converts extracted_span/train.txt to train_json.txt (json format) and extracted_span/val.txt to val_json.txt
import json

def main():
    input_file = open("extracted_span/train.txt", "r")
    counter = 0
    for line in input_file:
        span_object = {}
        fields = line.split("</s>")
        query = fields[0]
        query = query.strip()
        content = fields[1]
        content = content.strip()
        span_object['query'] = query
        span_object['answer'] = content
        json_to_write = json.dumps(span_object, indent = 4)
        output_file = open("extracted_span/jsons/train/" + str(counter) + ".json", "w")
        output_file.write(json_to_write)
        output_file.close()
        counter += 1
    
    input_file = open("extracted_span/val.txt", "r")
    for line in input_file:
        span_object = {}
        fields = line.split("</s>")
        query = fields[0]
        query = query.strip()
        content = fields[1]
        content = content.strip()
        span_object['query'] = query
        span_object['answer'] = content
        json_to_write = json.dumps(span_object, indent = 4)
        output_file = open("extracted_span/jsons/val/" + str(counter) + ".json", "w")
        output_file.write(json_to_write)
        output_file.close()
        counter += 1
    
    
        

if __name__ == "__main__":
    main()