import logging
import os
import zipfile

log = logging.getLogger(__name__)

path_to_cnndm_zip = "CNNDM/CNNDM.zip"
path_to_check_cnndm_already_extracted = "CNNDM/train"

def output(message):
    log.info(message)
    print(message)


def main():
    '''
        If not already present, this function will process the following files:
        a) Unzipping CNNDM.zip in folder CNNDM
        
    '''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(path_to_check_cnndm_already_extracted):
        output("CNNDM already extracted. Skipping this step")
    else:
        output("Starting to extract CNNDM.zip. This can take some minutes")
        with zipfile.ZipFile(path_to_cnndm_zip, 'r') as zip_ref:
            zip_ref.extractall(current_dir)
        output("CNNDM extracted")
    

if __name__ == "__main__":
    main()