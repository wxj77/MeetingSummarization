# Script to transform the json data in data/ALL/train|val into json format used by HMNET
# Usage: python transform_all_hmnet.py --mode=train to transform training data
# python transform_all_hmnet.py --mode=val for validation data
# As an additional parameter, --include_queries can be provided. If set to true, then queries will be included in the generated json
# and the json will be written into the .._hmnet_with_queries subfolders
# Additionall, the parameter --max_files can be passed (max. number of files to process, for testing purposes)

import argparse
import json
import os
import spacy

# You need to run once from the command line
# python -m spacy download en_core_web_sm

# Spacy part/code logic from
# https://github.com/microsoft/HMNet/issues/2
import spacy
nlp = spacy.load('en', parser = False)
POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}

def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s
    
def _parse_tags(parsed_text):
    output = {  'word': [],
                'pos_id': [],
                'ent_id': []}

    for token in parsed_text:
        #print(token)
        #[(token.text,token.idx) for token in parsed_sentence]
        output['word'].append(_str(token.text))
        pos = token.tag_
        output['pos_id'].append(POS[pos] if pos in POS else 0)

        ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
        output['ent_id'].append(ENT[ent] if ent in ENT else 0)

    word_idx = 0
    for sent in parsed_text.sents:
        # output['sentences'].append((word_idx, word_idx + len(sent)))
        word_idx += len(sent)

    # import pdb; pdb.set_trace()
    assert word_idx == len(output['word'])
    #print(output)
    if len(output['word']) == 0:
        return None
    assert len(output['word']) > 0

    return output

def create_speaker_mapping(meeting_transcripts):
    """ 
        Extract all unique speakers and map them to A, B, C etc and return the mapping
    """
    
    set_unique_speakers = set()
    speaker_mapping = {}
    for meeting_transcript in meeting_transcripts:
        speaker = meeting_transcript["speaker"]
        if speaker not in set_unique_speakers:
            set_unique_speakers.add(speaker)
    #print("Number speakers {}".format(len(set_unique_speakers)))
    current_asci = 64
    current_asci_text = ''
    for counter, speaker in enumerate(set_unique_speakers):
        #print(counter)
        current_asci += 1
        if current_asci == 91:
            current_asci = 65
            current_asci_text = "A"
        
        current_asci_char = chr(current_asci)
        speaker_char = current_asci_text + current_asci_char
        speaker_mapping[speaker] = speaker_char
    return speaker_mapping
    

def main():
    parser = argparse.ArgumentParser(
        description='determining whether to work on train or validation data'
    )
    parser.add_argument('--mode', type=str, required=True,
                        help='Please provide --mode arg with either train for training data or val for validation data')
    parser.add_argument('--include_queries', default=False,
                        help='Include topics/queries in generated json files')
    parser.add_argument('--max_files', type=int, action='store', default = 10000000, help="Maximum number of files to process")                    
    args = parser.parse_args()
    assert args.mode in ['train', 'val']
    
    if args.mode == "train" and args.include_queries == False:
        input_dir = "data/ALL/train"
        output_dir = "data/ALL/train_hmnet"
    elif args.mode == "train" and args.include_queries == True:
        raise NotImplementedError
        #input_dir = "data/ALL/train"
        #output_dir = "data/ALL/train_hmnet_with_queries"
    elif args.mode == "val" and args.include_queries == False:
        input_dir = "data/ALL/val"
        output_dir = "data/ALL/val_hmnet"
    else:
        raise NotImplementedError
        #input_dir = "data/ALL/val"
        #output_dir = "data/ALL/val_hmnet_with_queries"
    
    for counter, file_name in enumerate(os.listdir(input_dir)):
        if counter > args.max_files:
            break
        #print(file_name)
        meeting_id = file_name.replace(".json","")
        input_file = open(input_dir + "/" + file_name)
        file_content = input_file.read()
        file_content_json = json.loads(file_content)
        #speaker_mapping = create_speaker_mapping(file_content_json['meeting_transcripts'])
        #print(speaker_mapping)
        json_object_output = {}
        json_object_output["id"] = meeting_id
        meeting_array = []
        json_object_output["meeting"] = meeting_array
        meeting_transcripts = file_content_json['meeting_transcripts']
        for meeting_transcript in meeting_transcripts:
            meeting_json = {}
            speaker_name = meeting_transcript["speaker"]
            #print(speaker_name)
            role = speaker_name.split(" ")[0]
            try:
                speaker_encode = speaker_name.split(" ")[1]
            except:
                speaker_encode = ""
            meeting_json["speaker"] = speaker_encode
            meeting_json["role"] = role
            
            meeting_content = meeting_transcript["content"]
            processed_data_obj = _parse_tags(nlp(meeting_content))
            meeting_json["utt"] = processed_data_obj
            meeting_array.append(meeting_json)
        output_file = open(output_dir + "/" + file_name, "w")
        output_content = json.dumps(json_object_output, indent=2)
        output_file.write(output_content)
        


if __name__ == "__main__":
    main()