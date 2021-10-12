import logging
import os
import requests
import tempfile
import zipfile
from zipfile import ZipFile
from urllib.request import urlopen

log = logging.getLogger(__name__)

path_to_tmp_zip = "../../qmsum_data/tmp.zip"
extraction_path = "../../qmsum_data/CNNDM"
url_to_cnndm_zip = "https://meeting-summarization-project.s3.amazonaws.com/CNNDM.zip"
url_to_bert_bin = "https://meeting-summarization-project.s3.amazonaws.com/pytorch_model.bin"
path_to_bert_bin = "model/bert-large-uncased/pytorch_model.bin"
url_to_flax_model = "https://meeting-summarization-project.s3.amazonaws.com/flax_model.msgpack"
path_to_flax_model = "model/bert-large-uncased/flax_model.msgback"
url_to_tf_model = "https://meeting-summarization-project.s3.amazonaws.com/tf_model.h5"
path_to_tf_model = "model/bert-large-uncased/tf_model.h5"
url_to_whole_word_masking = "https://meeting-summarization-project.s3.amazonaws.com/whole-word-masking.tar.gz"
path_to_whole_word_masking = "model/bert-large-uncased/whole-word-masking.tar.gz"

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
    
    if os.path.exists(path_to_flax_model):
        output("File flax_model.msgback already exists in Bert large uncased model. Skipping this step")
    else:
        output("Downloading flax_model.msgpack for Bert large uncased. This can take some minutes.")
        url_file = urlopen(url_to_flax_model)
        file_output = open(path_to_flax_model, "wb")
        file_output.write(url_file.read())
        file_output.close()
        output("File flax_model.msgback written into models/bert-large-uncased.")
    
    if os.path.exists(path_to_bert_bin):
        output("File pytorch_model.bin already exists in Bert large uncased model. Skipping this step")
    else:
        output("Downloading pytorch_model.bin for Bert large uncased. This can take some minutes.")
        url_file = urlopen(url_to_bert_bin)
        file_output = open(path_to_bert_bin, "wb")
        file_output.write(url_file.read())
        file_output.close()
        output("File pytorch_model.bin written into models/bert-large-uncased.")
    
    if os.path.exists(path_to_tf_model):
        output("File tf_model.h5 already exists in Bert large uncased model. Skipping this step")
    else:
        output("Downloading tf_model.h5 for Bert large uncased. This can take some minutes.")
        url_file = urlopen(url_to_tf_model)
        file_output = open(path_to_tf_model, "wb")
        file_output.write(url_file.read())
        file_output.close()
        output("File tf_model.h5 written into models/bert-large-uncased.")
    
    if os.path.exists(path_to_whole_word_masking):
        output("File whole-word-masking.tar.gz already exists in Bert large uncased model. Skipping this step")
    else:
        output("Downloading whole-word-masking.tar.gz for Bert large uncased model")
        r = requests.get(url_to_whole_word_masking, stream=True)
        with open(path_to_whole_word_masking, 'wb') as f:
            for chunk in r.raw.stream(1024, decode_content=False):
                if chunk:
                    f.write(chunk)
        
        
        

if __name__ == "__main__":
    main()