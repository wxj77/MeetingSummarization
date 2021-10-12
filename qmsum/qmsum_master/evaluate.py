""" evaluation scripts"""
import re
import os
from os.path import join
import logging
import tempfile
import subprocess as sp

from cytoolz import curry

from pyrouge import Rouge155
from pyrouge.utils import log
from pathlib import Path

current_path = Path(os.getcwd())
parent_path = current_path.parent.absolute()
#parent_path = parent_path.parent.absolute()
#DATA_DIR = '../../../qmsum_data/CNNDM'
DATASET_DIR = os.path.join(parent_path, "qmsum_data", "CNNDM","CNNDM")
_ROUGE_PATH = os.path.join(parent_path, "qmsum_master","ROUGE-1.5.5")

#_ROUGE_PATH = 'venv/lib/python3.7/site-packages/rouge'

def eval_rouge(dec_dir, ref_dir):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
