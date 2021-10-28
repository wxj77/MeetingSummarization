import os
from utils.utils import compare_models
import logging
from datetime import datetime
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import AMIDataset
from data.dataset import CNNDataset
from data.dataset import OverallDataset
from models.model import SummarizationModel
from utils.checkpointing import CheckpointManager, load_checkpoint, dump_vocab
from predictor import Predictor
import sys


class Summarization(object):
    def __init__(self, hparams, mode='train'):
        print("Initializing summarization")
        #sys.exit()
        self.hparams = hparams
        self._logger = logging.getLogger(__name__)
        print('self.hparams:', self.hparams)
        self.logger = logging.getLogger(__name__)
        self.model_name = self.hparams.model_name

        if hparams.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
       
        self.build_dataloader_overall()
        #sys.exit()
        self.build_dataloader_cnn()
        self.build_dataloader()
        print("DAtaloaders finished")
        
        # Merge self.vocab_word_cnn, self.vocab_word_qmsum
        

        self.save_dirpath = self.model_name
        #self.save_dirpath = self.save_dirpath
        today = str(datetime.today().month) + 'M_' + str(datetime.today().day) + 'D' + '_GEN_MAX_' + str(
            self.hparams.gen_max_length)
        #tensorboard_path = self.save_dirpath + today
        tensorboard_path = self.save_dirpath
        print("Tensorboard path")
        print(tensorboard_path)
        #sys.exit()
        self.summary_writer = SummaryWriter(tensorboard_path, comment="Unmt")

        if mode == 'train':
            #sys.exit()
            self.build_model()
            self.setup_training()
            self.predictor = self.build_eval_model(model=self.model, summary_writer=self.summary_writer)
            dump_vocab(self.save_dirpath + 'vocab_word', self.vocab_word, self.save_dirpath + "/" + 'vocab_word')

        elif mode == 'eval':
            self.predictor = self.build_eval_model(summary_writer=self.summary_writer)

    def build_dataloader(self):
        print("Batch size")
        print(self.hparams.batch_size)
        self.train_dataset = AMIDataset(self.hparams, type='train')
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        self.vocab_word_qmsum = self.train_dataset.vocab_word
        self.vocab_role_qmsum = self.train_dataset.vocab_role
        self.vocab_pos_qmsum = self.train_dataset.vocab_pos

        self.test_dataset = AMIDataset(self.hparams, type='test',
                                       vocab_word=self.vocab_word, vocab_role=self.vocab_role, vocab_pos=self.vocab_pos)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            drop_last=False
        )
    
    def build_dataloader_overall(self):
        
        print("Batch size")
        print(self.hparams.batch_size)
        self.train_dataset_overall = OverallDataset(self.hparams, type='train')
        #sys.exit()
        self.train_dataloader_overall = DataLoader(
            self.train_dataset_overall,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        self.vocab_word = self.train_dataset_overall.vocab_word
        print("Len vocab word")
        print(len(self.vocab_word))
        self.vocab_role = self.train_dataset_overall.vocab_role
        self.vocab_pos = self.train_dataset_overall.vocab_pos

        

    print("""
           # -------------------------------------------------------------------------
           #   DATALOADER Overall FINISHED
           # -------------------------------------------------------------------------
           """)
           
    def build_dataloader_cnn(self):
        print("Batch size")
        print(self.hparams.batch_size)
        self.train_dataset_cnn = CNNDataset(self.hparams, type='train')
        self.train_dataloader_cnn = DataLoader(
            self.train_dataset_cnn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            shuffle=True,
            drop_last=True
        )
        self.vocab_word_cnn = self.train_dataset_cnn.vocab_word
        #print(self.vocab_word_cnn.token2id)
        #sys.exit()
        #print("self.vocab_word shape")
        #print(type(self.vocab_word))
        #print("self.vocab_word keys")
        #for key in self.vocab_word:
        #    print(key)
        #print(self.vocab_word.id2token)
        #sys.exit()
        self.vocab_role_cnn = self.train_dataset_cnn.vocab_role
        self.vocab_pos_cnn = self.train_dataset_cnn.vocab_pos

       

    print("""
           # -------------------------------------------------------------------------
           #   DATALOADER CCN FINISHED
           # -------------------------------------------------------------------------
           """)       

    def build_model(self):
        # Define model
        print("Model parameters")
        print("Vocab word")
        print(len(self.vocab_word))
        self.model = SummarizationModel(hparams=self.hparams, vocab_word=self.vocab_word,
                                        vocab_role=self.vocab_role, vocab_pos=self.vocab_pos)

        # Multi-GPU
        self.model = self.model.to(self.device)

        # Use Multi-GPUs
        if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

        # Define Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, betas=(self.hparams.optimizer_adam_beta1,
                                                                                               self.hparams.optimizer_adam_beta2))

    def setup_training(self):
        self.save_dirpath = self.hparams.save_dirpath
        today = str(datetime.today().month) + 'M_' + str(datetime.today().day) + 'D'
        tensorboard_path = self.save_dirpath + today
        self.summary_writer = SummaryWriter(tensorboard_path, comment="Unmt")
        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer,
                                                    self.save_dirpath, hparams=self.hparams)

        # If loading from checkpoint, adjust start epoch and load parameters.
        if self.hparams.load_pthpath == "":
            self.start_epoch = 1
        else:
            # "path/to/checkpoint_xx.pth" -> xx
            self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
            self.start_epoch += 1
            model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=True)
            else:
                self.model.load_state_dict(model_state_dict)

            self.optimizer.load_state_dict(optimizer_state_dict, strict=True)
            self.previous_model_path = self.hparams.load_pthpath
            print("Loaded model from {}".format(self.hparams.load_pthpath))

        print(
            """
            # -------------------------------------------------------------------------
            #   Setup Training Finished
            # -------------------------------------------------------------------------
            """
        )

    def build_eval_model(self, model=None, summary_writer=None, eval_path=None):
        # Define predictor
        predictor = Predictor(self.hparams, model=model, vocab_word=self.vocab_word,
                                   vocab_role=self.vocab_role, vocab_pos=self.vocab_pos,
                                   checkpoint=eval_path, summary_writer=summary_writer)

        return predictor

    def train(self):
        train_begin = datetime.utcnow()  # News
        global_iteration_step = 0
        # Begin Pre-Training
        for epoch in range(self.hparams.num_epochs_cnn):
            print("Epoch {}".format(epoch))
            batches_processed = 0
            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader_cnn)
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                #if batches_processed + 8 > self.hparams.max_batch_numbers_per_epoch:
                #    break
                #print(len(batch))
                data = batch
                if 'dialogues_ids' not in data:
                    continue
                dialogues_ids = data['dialogues_ids'].to(self.device)
                pos_ids = data['pos_ids'].to(self.device)
                labels_ids = data['labels_ids'].to(self.device) # [batch==1, tgt_seq_len]
                src_masks = data['src_masks'].to(self.device)
                role_ids = data['role_ids'].to(self.device)
                
                #print("shapes CNN dataset")
                #print("dialogue_ids")
                #print(dialogues_ids.shape)
                #print("labels_ids")
                #print(labels_ids.shape)
                #print("src makes")
                #print(src_masks.shape)
                #print("role_ids")
                #print(role_ids.shape)
                #print("pos_ids")
                #print(pos_ids.shape)

                logits = self.model(inputs=dialogues_ids, targets=labels_ids[:, :-1],  # before <END> token
                                    src_masks=src_masks, role_ids=role_ids, pos_ids=pos_ids) # [batch x tgt_seq_len, vocab_size]

                labels_ids = labels_ids[:, 1:]
                labels_ids = labels_ids.view(labels_ids.shape[0] * labels_ids.shape[1]) # [batch x tgt_seq_len]

                loss = self.criterion(logits, labels_ids)
                loss.backward()

                # gradient cliping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                global_iteration_step += 1
                description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                    datetime.utcnow() - train_begin,
                    epoch,
                    global_iteration_step, loss,
                    self.optimizer.param_groups[0]['lr'])
                tqdm_batch_iterator.set_description(description)
                batches_processed += 1
        
        for epoch in range(self.hparams.num_epochs):
            print("Epoch {}".format(epoch))
            batches_processed = 0
            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            for batch_idx, batch in enumerate(tqdm_batch_iterator):
                #if batches_processed + 8 > self.hparams.max_batch_numbers_per_epoch:
                #    break
                #print(len(batch))
                data = batch
                #print("data")
                #print(data)
                if 'dialogues_ids' not in data:
                    continue
                dialogues_ids = data['dialogues_ids'].to(self.device)
                pos_ids = data['pos_ids'].to(self.device)
                labels_ids = data['labels_ids'].to(self.device) # [batch==1, tgt_seq_len]
                src_masks = data['src_masks'].to(self.device)
                role_ids = data['role_ids'].to(self.device)
                
                #print("shapes AMI dataset")
                #print("dialogue_ids")
                #print(dialogues_ids.shape)
                #print("labels_ids")
                #print(labels_ids.shape)
                #print("src makes")
                #print(src_masks.shape)
                #print("role_ids")
                #print(role_ids.shape)
                #print("pos_ids")
                #print(pos_ids.shape)
                
                logits = self.model(inputs=dialogues_ids, targets=labels_ids[:, :-1],  # before <END> token
                                    src_masks=src_masks, role_ids=role_ids, pos_ids=pos_ids) # [batch x tgt_seq_len, vocab_size]

                labels_ids = labels_ids[:, 1:]
                labels_ids = labels_ids.view(labels_ids.shape[0] * labels_ids.shape[1]) # [batch x tgt_seq_len]

                loss = self.criterion(logits, labels_ids)
                loss.backward()

                # gradient cliping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                #except:
                 #   pass

                global_iteration_step += 1
                description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                    datetime.utcnow() - train_begin,
                    epoch,
                    global_iteration_step, loss,
                    self.optimizer.param_groups[0]['lr'])
                tqdm_batch_iterator.set_description(description)
                batches_processed += 1

            # # -------------------------------------------------------------------------
            # #   ON EPOCH END  (checkpointing and validation)
            # # -------------------------------------------------------------------------
            self.checkpoint_manager.step(epoch)
            self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
            self._logger.info(self.previous_model_path)

            # torch.cuda.empty_cache()

            if epoch % 10 == 0 and epoch >= self.hparams.start_eval_epoch:
                print('======= Evaluation Start Epoch: ', epoch, ' ==================')

                self.predictor.evaluate(test_dataloader=self.test_dataloader, epoch=epoch,
                                        eval_path=self.previous_model_path)

                print('============================================================\n\n')