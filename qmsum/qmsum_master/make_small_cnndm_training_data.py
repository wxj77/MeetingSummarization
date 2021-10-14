# Executing this script will only keep x training datasets in CNNDM folder
# The script assumes that the folder CNNDM already exists
# Useful to quickly test changes to existing models
# Usage example: python make_small_cnndm_training.py 10000 # Will only keep random 10k training data
# The script will first make a copy of the current CNNDM train folder so that the original state can be reverted fast

import os
import sys
import shutil

from pathlib import Path

def main():
    number_of_training_data_to_keep = sys.argv[1]
    number_of_training_data_to_keep = int(number_of_training_data_to_keep)
    current_path = Path(os.getcwd())
    parent_path = current_path.parent.absolute()
    parent_path = parent_path.parent.absolute()
    parent_path = parent_path.parent.absolute()
    src = os.path.join(parent_path, "MeetingSummarization","qmsum_data", "CNNDM", "CNNDM", "train")
    dst = os.path.join(parent_path, "MeetingSummarization","qmsum_data", "CNNDM", "CNNDM", "train_original")
    if not os.path.exists(dst):
        os.makedirs(dst)
    print("Saving original contents of train")
    for file_name in os.listdir(src):
    # construct full file path
        #print(file_name)
        source = src + "/" + file_name
        destination = dst + "/" + file_name
    # copy only files
        
        shutil.copy(source, destination)
        #print('copied', file_name)
    
    print("Original contents of train saved")
    print("Starting to reduce size of original train directory")
    for file_name in os.listdir(src):
        #print(file_name)
        fields = file_name.split(".")
        json_number = int(fields[0])
        source = src + "/" + file_name
        if json_number > number_of_training_data_to_keep + 2:
            os.remove(source)
    
    print("Kept {} training data in training directory. You can find all original training data in the folder train_original".format(number_of_training_data_to_keep))

if __name__ == "__main__":
    main()