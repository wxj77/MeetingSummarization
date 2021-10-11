import logging
import os
import zipfile
from zipfile import ZipFile
from urllib.request import urlopen

log = logging.getLogger(__name__)

path_to_tmp_zip = "../../qmsum_data/tmp.zip"
extraction_path = "../../qmsum_data/CNNDM"
url_to_cnndm_zip = "https://meeting-summarization-project.s3.amazonaws.com/CNNDM.zip"
url_to_bert_large_uncased = "https://meeting-summarization-project.s3.amazonaws.com/pytorch_bert_large_uncased.bin"
path_bert_large_uncased = "model/pytorch_bert_large_uncased.bin"

def output(message):
    log.info(message)
    print(message)


def main():
    '''
        If not already present, this function will process the following files:
        a) Unzipping CNNDM.zip in folder qmsum_data/CNNDM
        b) Downloading BERT large uncased model and saving it in folder model in qmsum_master
        
    '''
    if os.path.exists("../../qmsum_data/CNNDM"):
        output("CNNDM data already extracted. Skipping this step")
    else:
        output("Starting to download CNNDM.zip. This can take some minutes.")
        zipresp = urlopen(url_to_cnndm_zip)
        tempzip = open(path_to_tmp_zip, "wb")
        tempzip.write(zipresp.read())
        tempzip.close()
        zf = ZipFile(path_to_tmp_zip)
        zf.extractall(path=extraction_path)
        zf.close()
        output("CNNDM extracted")
    
    if os.path.exists(path_bert_large_uncased):
        output("Bert uncased large model already in subfolder model. Skipping this step")
    else:
        output("Downloading Bert large uncased model for pytorch. This can take some minutes.")
        bert_model = urlopen(url_to_bert_large_uncased)
        bert_output = open(path_bert_large_uncased, "wb")
        bert_output.write(bert_model.read())
        bert_output.close()
        output("Bert large uncased for Pytorch written in model folder.")
        
        
        

if __name__ == "__main__":
    main()