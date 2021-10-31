import torch
from torch.utils.data import Dataset
from models.transformer.layers import _gen_seq_bias_mask
from collections import Counter
from tqdm import tqdm
import sys

# For input dialogues
PAD = 0
BOS = 1
EOS = 2
UNK = 3
# For target summaries
BEGIN = 4
END = 5

vocab_word_overall = None
vocab_pos_overall = None
vocab_role_overall = None


class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class OverallDataset(Dataset):
    def __init__(self, hparams, type='', vocab_word=None,
                 vocab_role=None, vocab_pos=None, max_vocab_size=100000):
        super().__init__()
        self.hparams = hparams
        global vocab_word_overall
        global vocab_pos_overall
        global vocab_role_overall

        self.input_examples = torch.load(hparams.data_dir + type + "_qmsum_data_queries"+'_corpus')
        self.input_examples_cnn = torch.load("data/train_cnn_corpus")
        #self.input_examples = torch.load('data/train_cnn_corpus')
        #self.input_examples = torch.load(hparams.data_dir + type + "_qmsum_data_queries"+'_corpus')
        print('[%s] %d examples is loaded' % (type, len(self.input_examples)))

        self.data_list = []
        for key, value in self.input_examples.items():
            #print("key {}".format(key))
            #print("value {}".format(value))
            texts = value['texts']
            labels = value['labels']
            #query = value['query']
            #query = value['query']
            #print(query)
            dialogues = []
            #query_sentence = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_sentence = query_sentence.strip().lower()
            #query_pos = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_pos = query_pos.strip().lower()
            #dialogues.append({'role': 'questioner', 'sentence': query_sentence, 'pos_sentence': query_pos})
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                sentence = sentence.strip()
                pos_sentence = pos_sentence.strip()
                role = role.strip()
                if role != "" and sentence != "" and pos_sentence != "":
                    dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            self.data_list.append({'labels': labels, 'dialogues': dialogues})

        #if (vocab_word == None) and (vocab_role == None):
        #    counter, role_counter, pos_counter = self.build_counter()
        #    self.vocab_word = self.build_vocab(counter, max_vocab_size, type='word')
        #    self.vocab_role = self.build_vocab(role_counter, max_vocab_size, type='role')
        #    self.vocab_pos = self.build_vocab(pos_counter, max_vocab_size, type='pos')
        #else:
        #    self.vocab_word = vocab_word
        #    self.vocab_role = vocab_role
        #    self.vocab_pos = vocab_pos
        
        for key, value in self.input_examples_cnn.items():
            #print("key {}".format(key))
            #print("value {}".format(value))
            texts = value['texts']
            labels = value['labels']
            #query = value['query']
            #query = value['query']
            #print(query)
            dialogues = []
            #query_sentence = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_sentence = query_sentence.strip().lower()
            #query_pos = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_pos = query_pos.strip().lower()
            #dialogues.append({'role': 'questioner', 'sentence': query_sentence, 'pos_sentence': query_pos})
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                sentence = sentence.strip()
                pos_sentence = pos_sentence.strip()
                role = role.strip()
                if role != "" and sentence != "" and pos_sentence != "":
                    dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            self.data_list.append({'labels': labels, 'dialogues': dialogues})
        
        #sys.exit()
        if (vocab_word == None) and (vocab_role == None):
            counter, role_counter, pos_counter = self.build_counter()
            #sys.exit()
            self.vocab_word = self.build_vocab(counter, max_vocab_size, type='word')
            self.vocab_role = self.build_vocab(role_counter, max_vocab_size, type='role')
            self.vocab_pos = self.build_vocab(pos_counter, max_vocab_size, type='pos')
            vocab_word_overall = self.vocab_word
            vocab_role_overall = self.vocab_role
            vocab_pos_overall = self.vocab_pos
        else:
            self.vocab_word = vocab_word_overall
            self.vocab_role = vocab_role_overall
            self.vocab_pos = vocab_pos_overall
            vocab_word_overall = self.vocab_word
            vocab_role_overall = self.vocab_role
            vocab_pos_overall = self.vocab_pos

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """

        :param index:
        :return:
            Input Examples (vocab_ids)
                dialogues: list of (role, utterance)
                labels: reference summaries for texts
        """
        dialogues = self.data_list[index]['dialogues']
        labels = self.data_list[index]['labels']

        dialogues_ids = [] # (num_turn, seq_len)
        pos_ids = [] # (num_turn, seq_len)
        role_ids = [] # (num_turns)

        label_tokens = self.tokenize(labels)

        labels_ids = self.tokens2ids(label_tokens, self.vocab_word.token2id,
                                     is_reference=True) #(seq_len)
        
        #print("Dialogue 17")
        #print(dialogues[17])
        for turn_idx, dialogue in enumerate(dialogues):
            #print('turn_idx: ')
            #print(turn_idx)
            #print("dialogue")
            #print(dialogue)
            if turn_idx >= self.hparams.max_length:
                break
            sentence = dialogue['sentence']
            sentence_count = len(sentence.split('.')) - 1
            if sentence_count < 2 or len(sentence) < 4:
                continue
            sentence = sentence.replace('. .', '.').replace(', ,', ',')

            tokens = self.tokenize(sentence)
            #print("dialogue")
            #print(dialogue)
            #print("sentence")
            #print(sentence)
            #print("tokens")
            #print(tokens)
            pos_tokens = self.tokenize(dialogue['pos_sentence'])
            role_tokens = self.tokenize(dialogue['role'])

            if len(tokens) >= self.hparams.max_length - 2:
                tokens = tokens[:self.hparams.max_length - 2]
                pos_tokens = pos_tokens[:self.hparams.max_length - 2]

            token_ids = self.tokens2ids(tokens, self.vocab_word.token2id)
            pos_token_ids = self.tokens2ids(pos_tokens, self.vocab_pos.token2id)
            role_token_ids = self.tokens2ids(role_tokens, self.vocab_role.token2id, is_role=True)
            
            #if len(token_ids) == 0 or len(pos_token_ids) == 0 or len(role_token_ids) == 0:
            #    print("empty lists detected")
            #if len(token_ids) > 0 and len(pos_token_ids) > 0 and len(role_token_ids) > 0:
            dialogues_ids.append(token_ids)
            pos_ids.append(pos_token_ids)
            role_ids.append(role_token_ids)
        #print(dialogues_ids)
        #print("dialogue_ids")
        #print(dialogues_ids)
        #print("padding dialogue ids")
        #print(dialogues_ids)
        try:
            padded_dialogues, dialogues_lens, src_masks = self.pad_sequence(dialogues_ids, "dialogues")
            padded_pos_ids, _, _ = self.pad_sequence(pos_ids, "positions")
        
            for counter, role_id in enumerate(role_ids):
            #print(role_id)
                if len(role_id) != 9:
                    role_id = [3, 3, 3, 3, 3, 3, 3, 3, 3]
                    role_ids[counter] = role_id

            data = dict()
            data['dialogues'] = dialogues
            data['labels'] = labels
            data['dialogues_ids'] = padded_dialogues
            data['pos_ids'] = padded_pos_ids
            data['dialogues_lens'] = torch.tensor(dialogues_lens).long()
            data['src_masks'] = src_masks
        #print(role_ids)

            data['role_ids'] = torch.tensor(role_ids).long()
            data['labels_ids'] = torch.tensor(labels_ids).long()
            return data
        except ValueError:
            data = dict()
            return data

    def pad_sequence(self, seqs, what=''):
        #print("Padding seqs {}".format(what))
        #print(seqs)
        lens = [len(seq) for seq in seqs]
        #print("lens")
        #print(lens)
        max_seq_length = max(lens)
        #print("max seq length")
        #print(max_seq_length)
        
        padded_seqs = torch.zeros(len(seqs), max_seq_length).long()
        
        for i, seq in enumerate(seqs):
            end_idx = lens[i]
            padded_seqs[i, :end_idx] = torch.LongTensor(seq[:end_idx])

        src_masks = _gen_seq_bias_mask(lens, max_seq_length)

        return padded_seqs, lens, src_masks

    def tokenize(self, sentence):
        return sentence.split()

    def build_counter(self):
        counter = Counter()
        role_counter = Counter()
        pos_counter = Counter()

        role_words = []
        sentence_words = []
        pos_words = []

        augmented_data_list = [] # add dev vocab
        augmented_data_list.extend(self.data_list)

        dev_examples = torch.load(self.hparams.data_dir + 'dev_corpus')
        #sys.exit()
        print('[%s] %d examples is loaded' % ('Dev', len(dev_examples)))
        for key, value in dev_examples.items():
            texts = value['texts']
            labels = value['labels']
            dialogues = []
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[1] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            augmented_data_list.append({'labels': labels, 'dialogues': dialogues})

        for data in augmented_data_list:
            dialogues = data['dialogues']
            for dialogue in dialogues:
                sentence_words.append(self.tokenize(dialogue['sentence']))
                pos_words.append(self.tokenize(dialogue['pos_sentence']))
                role_words.append([dialogue['role']])
            labels = data['labels']
            sentence_words.append(self.tokenize(labels))

        for words in sentence_words:
            counter.update(words)

        for role in role_words:
            role_counter.update(role)

        for pos in pos_words:
            pos_counter.update(pos)
        #sys.exit()
        #print('role_counter: ', role_counter)
        #print('pos_counter: ', pos_counter)

        return counter, role_counter, pos_counter

    def build_vocab(self, counter, max_vocab_size, type='word'):
        vocab = AttrDict()
        if type == 'word':
            print("\n===== Building [Word Vocab] =========")
        elif type == 'role':
            print("\n===== Building [Role Vocab] =========")
        elif type == 'pos':
            print("\n===== Building [POS Vocab] =========")

        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS,
                          '<UNK>': UNK, '<BEGIN>': BEGIN, '<END>': END}
        preset_vocab_size = len(vocab.token2id)
        print('preset_vocab_size: ', preset_vocab_size)
        vocab.token2id.update(
            {token: _id + preset_vocab_size for _id, (token, count) in
             tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        print('Vocab size: ', len(vocab.token2id))
        print('==========================================')
        return vocab

    def tokens2ids(self, tokens, token2id, is_reference=False, is_role=False):
        seq = []
        if is_reference is False and is_role is False:
            # For input dialogues.
            seq.append(BOS)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(EOS)
        elif is_reference:
            # For target summaries.
            seq.append(BEGIN)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(END)

        if is_role:
            seq.extend([token2id.get(token, UNK) for token in tokens])
        return seq

class AMIDataset(Dataset):
    def __init__(self, hparams, type='', vocab_word=None,
                 vocab_role=None, vocab_pos=None, max_vocab_size=50000):
        super().__init__()
        self.hparams = hparams
        global vocab_word_overall
        global vocab_pos_overall
        global vocab_role_overall

        #self.input_examples = torch.load(hparams.data_dir + type + '_corpus')
        #self.input_examples = torch.load('data/train_cnn_corpus')
        self.input_examples = torch.load(hparams.data_dir + type + "_qmsum_data_queries"+'_corpus')
        print('[%s] %d examples is loaded' % (type, len(self.input_examples)))

        self.data_list = []
        for key, value in self.input_examples.items():
            #print("key {}".format(key))
            #print("value {}".format(value))
            texts = value['texts']
            labels = value['labels']
            #query = value['query']
            #query = value['query']
            #print(query)
            dialogues = []
            #query_sentence = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_sentence = query_sentence.strip().lower()
            #query_pos = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_pos = query_pos.strip().lower()
            #dialogues.append({'role': 'questioner', 'sentence': query_sentence, 'pos_sentence': query_pos})
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                sentence = sentence.strip()
                pos_sentence = pos_sentence.strip()
                role = role.strip()
                if role != "" and sentence != "" and pos_sentence != "":
                    dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            self.data_list.append({'labels': labels, 'dialogues': dialogues})
        
        self.vocab_word = vocab_word_overall
        self.vocab_pos = vocab_pos_overall
        self.vocab_role = vocab_role_overall
        
        #if (vocab_word == None) and (vocab_role == None):
        #    counter, role_counter, pos_counter = self.build_counter()
        #    self.vocab_word = self.build_vocab(counter, max_vocab_size, type='word')
        #    self.vocab_role = self.build_vocab(role_counter, max_vocab_size, type='role')
        #    self.vocab_pos = self.build_vocab(pos_counter, max_vocab_size, type='pos')
        #else:
        #    self.vocab_word = vocab_word
        #    self.vocab_role = vocab_role
        #    self.vocab_pos = vocab_pos

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """

        :param index:
        :return:
            Input Examples (vocab_ids)
                dialogues: list of (role, utterance)
                labels: reference summaries for texts
        """
        dialogues = self.data_list[index]['dialogues']
        labels = self.data_list[index]['labels']

        dialogues_ids = [] # (num_turn, seq_len)
        pos_ids = [] # (num_turn, seq_len)
        role_ids = [] # (num_turns)

        label_tokens = self.tokenize(labels)

        labels_ids = self.tokens2ids(label_tokens, self.vocab_word.token2id,
                                     is_reference=True) #(seq_len)
        
        #print("Dialogue 17")
        #print(dialogues[17])
        for turn_idx, dialogue in enumerate(dialogues):
            #print('turn_idx: ')
            #print(turn_idx)
            #print("dialogue")
            #print(dialogue)
            if turn_idx >= self.hparams.max_length:
                break
            sentence = dialogue['sentence']
            sentence_count = len(sentence.split('.')) - 1
            if sentence_count < 2 or len(sentence) < 4:
                continue
            sentence = sentence.replace('. .', '.').replace(', ,', ',')

            tokens = self.tokenize(sentence)
            #print("dialogue")
            #print(dialogue)
            #print("sentence")
            #print(sentence)
            #print("tokens")
            #print(tokens)
            pos_tokens = self.tokenize(dialogue['pos_sentence'])
            role_tokens = self.tokenize(dialogue['role'])

            if len(tokens) >= self.hparams.max_length - 2:
                tokens = tokens[:self.hparams.max_length - 2]
                pos_tokens = pos_tokens[:self.hparams.max_length - 2]

            token_ids = self.tokens2ids(tokens, self.vocab_word.token2id)
            pos_token_ids = self.tokens2ids(pos_tokens, self.vocab_pos.token2id)
            role_token_ids = self.tokens2ids(role_tokens, self.vocab_role.token2id, is_role=True)
            
            #if len(token_ids) == 0 or len(pos_token_ids) == 0 or len(role_token_ids) == 0:
            #    print("empty lists detected")
            #if len(token_ids) > 0 and len(pos_token_ids) > 0 and len(role_token_ids) > 0:
            dialogues_ids.append(token_ids)
            pos_ids.append(pos_token_ids)
            role_ids.append(role_token_ids)
        #print(dialogues_ids)
        #print("dialogue_ids")
        #print(dialogues_ids)
        #print("padding dialogue ids")
        #print(dialogues_ids)
        try:
            padded_dialogues, dialogues_lens, src_masks = self.pad_sequence(dialogues_ids, "dialogues")
            padded_pos_ids, _, _ = self.pad_sequence(pos_ids, "positions")
        
            for counter, role_id in enumerate(role_ids):
            #print(role_id)
                if len(role_id) != 9:
                    role_id = [3, 3, 3, 3, 3, 3, 3, 3, 3]
                    role_ids[counter] = role_id

            data = dict()
            data['dialogues'] = dialogues
            data['labels'] = labels
            data['dialogues_ids'] = padded_dialogues
            data['pos_ids'] = padded_pos_ids
            data['dialogues_lens'] = torch.tensor(dialogues_lens).long()
            data['src_masks'] = src_masks
        #print(role_ids)

            data['role_ids'] = torch.tensor(role_ids).long()
            data['labels_ids'] = torch.tensor(labels_ids).long()
            return data
        except ValueError:
            data = dict()
            return data

    def pad_sequence(self, seqs, what=''):
        #print("Padding seqs {}".format(what))
        #print(seqs)
        lens = [len(seq) for seq in seqs]
        #print("lens")
        #print(lens)
        max_seq_length = max(lens)
        #print("max seq length")
        #print(max_seq_length)
        
        padded_seqs = torch.zeros(len(seqs), max_seq_length).long()
        
        for i, seq in enumerate(seqs):
            end_idx = lens[i]
            padded_seqs[i, :end_idx] = torch.LongTensor(seq[:end_idx])

        src_masks = _gen_seq_bias_mask(lens, max_seq_length)

        return padded_seqs, lens, src_masks

    def tokenize(self, sentence):
        return sentence.split()

    def build_counter(self):
        counter = Counter()
        role_counter = Counter()
        pos_counter = Counter()

        role_words = []
        sentence_words = []
        pos_words = []

        augmented_data_list = [] # add dev vocab
        augmented_data_list.extend(self.data_list)

        dev_examples = torch.load(self.hparams.data_dir + 'dev_corpus')

        print('[%s] %d examples is loaded' % ('Dev', len(dev_examples)))
        for key, value in dev_examples.items():
            texts = value['texts']
            labels = value['labels']
            dialogues = []
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[1] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            augmented_data_list.append({'labels': labels, 'dialogues': dialogues})

        for data in augmented_data_list:
            dialogues = data['dialogues']
            for dialogue in dialogues:
                sentence_words.append(self.tokenize(dialogue['sentence']))
                pos_words.append(self.tokenize(dialogue['pos_sentence']))
                role_words.append([dialogue['role']])
            labels = data['labels']
            sentence_words.append(self.tokenize(labels))

        for words in sentence_words:
            counter.update(words)

        for role in role_words:
            role_counter.update(role)

        for pos in pos_words:
            pos_counter.update(pos)
        #print('role_counter: ', role_counter)
        #print('pos_counter: ', pos_counter)

        return counter, role_counter, pos_counter

    def build_vocab(self, counter, max_vocab_size, type='word'):
        vocab = AttrDict()
        if type == 'word':
            print("\n===== Building [Word Vocab] =========")
        elif type == 'role':
            print("\n===== Building [Role Vocab] =========")
        elif type == 'pos':
            print("\n===== Building [POS Vocab] =========")

        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS,
                          '<UNK>': UNK, '<BEGIN>': BEGIN, '<END>': END}
        preset_vocab_size = len(vocab.token2id)
        print('preset_vocab_size: ', preset_vocab_size)
        vocab.token2id.update(
            {token: _id + preset_vocab_size for _id, (token, count) in
             tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        print('Vocab size: ', len(vocab.token2id))
        print('==========================================')
        return vocab

    def tokens2ids(self, tokens, token2id, is_reference=False, is_role=False):
        seq = []
        if is_reference is False and is_role is False:
            # For input dialogues.
            seq.append(BOS)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(EOS)
        elif is_reference:
            # For target summaries.
            seq.append(BEGIN)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(END)

        if is_role:
            seq.extend([token2id.get(token, UNK) for token in tokens])
        return seq

class CNNDataset(Dataset):
    def __init__(self, hparams, type='', vocab_word=None,
                 vocab_role=None, vocab_pos=None, max_vocab_size=50000):
        super().__init__()
        self.hparams = hparams
        global vocab_word_overall
        global vocab_pos_overall
        global vocab_role_overall
        print("Len vocab overall init CNN {}".format(len(vocab_word_overall['token2id'])))
        print("Len pos overall init CNN {}".format(len(vocab_pos_overall['token2id'])))

        #self.input_examples = torch.load(hparams.data_dir + type + '_corpus')
        self.input_examples = torch.load('data/train_cnn_corpus')
        #self.input_examples = torch.load(hparams.data_dir + type + "_qmsum_data_queries"+'_corpus')
        print('[%s] %d examples is loaded' % (type, len(self.input_examples)))

        self.data_list = []
        for key, value in self.input_examples.items():
            #print("key {}".format(key))
            #print("value {}".format(value))
            texts = value['texts']
            labels = value['labels']
            #query = value['query']
            #query = value['query']
            #print(query)
            dialogues = []
            #query_sentence = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_sentence = query_sentence.strip().lower()
            #query_pos = ' '.join(word_pos.split('/')[0] for word_pos in query.split())
            #query_pos = query_pos.strip().lower()
            #dialogues.append({'role': 'questioner', 'sentence': query_sentence, 'pos_sentence': query_pos})
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                sentence = sentence.strip()
                pos_sentence = pos_sentence.strip()
                role = role.strip()
                if role != "" and sentence != "" and pos_sentence != "":
                    dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            self.data_list.append({'labels': labels, 'dialogues': dialogues})
        
        self.vocab_word = vocab_word_overall
        self.vocab_pos = vocab_pos_overall
        self.vocab_role = vocab_role_overall
        print("Len vocab local init CNN {}".format(len(self.vocab_word['token2id'])))

       # if (vocab_word == None) and (vocab_role == None):
    #        counter, role_counter, pos_counter = self.build_counter()
     #       self.vocab_word = self.build_vocab(counter, max_vocab_size, type='word')
     #       self.vocab_role = self.build_vocab(role_counter, max_vocab_size, type='role')
     #       self.vocab_pos = self.build_vocab(pos_counter, max_vocab_size, type='pos')
     #   else:
     #       self.vocab_word = vocab_word
     #       self.vocab_role = vocab_role
     #       self.vocab_pos = vocab_pos

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """

        :param index:
        :return:
            Input Examples (vocab_ids)
                dialogues: list of (role, utterance)
                labels: reference summaries for texts
        """
        dialogues = self.data_list[index]['dialogues']
        labels = self.data_list[index]['labels']

        dialogues_ids = [] # (num_turn, seq_len)
        pos_ids = [] # (num_turn, seq_len)
        role_ids = [] # (num_turns)

        label_tokens = self.tokenize(labels)

        labels_ids = self.tokens2ids(label_tokens, self.vocab_word.token2id,
                                     is_reference=True) #(seq_len)
        
        #print("Dialogue 17")
        #print(dialogues[17])
        for turn_idx, dialogue in enumerate(dialogues):
            #print('turn_idx: ')
            #print(turn_idx)
            #print("dialogue")
            #print(dialogue)
            if turn_idx >= self.hparams.max_length:
                break
            sentence = dialogue['sentence']
            sentence_count = len(sentence.split('.')) - 1
            if sentence_count < 2 or len(sentence) < 4:
                continue
            sentence = sentence.replace('. .', '.').replace(', ,', ',')

            tokens = self.tokenize(sentence)
            #print("dialogue")
            #print(dialogue)
            #print("sentence")
            #print(sentence)
            #print("tokens")
            #print(tokens)
            pos_tokens = self.tokenize(dialogue['pos_sentence'])
            role_tokens = self.tokenize(dialogue['role'])

            if len(tokens) >= self.hparams.max_length - 2:
                tokens = tokens[:self.hparams.max_length - 2]
                pos_tokens = pos_tokens[:self.hparams.max_length - 2]

            token_ids = self.tokens2ids(tokens, self.vocab_word.token2id)
            pos_token_ids = self.tokens2ids(pos_tokens, self.vocab_pos.token2id)
            role_token_ids = self.tokens2ids(role_tokens, self.vocab_role.token2id, is_role=True)
            
            #if len(token_ids) == 0 or len(pos_token_ids) == 0 or len(role_token_ids) == 0:
            #    print("empty lists detected")
            #if len(token_ids) > 0 and len(pos_token_ids) > 0 and len(role_token_ids) > 0:
            dialogues_ids.append(token_ids)
            pos_ids.append(pos_token_ids)
            role_ids.append(role_token_ids)
        #print(dialogues_ids)
        #print("dialogue_ids")
        #print(dialogues_ids)
        #print("padding dialogue ids")
        #print(dialogues_ids)
        try:
            padded_dialogues, dialogues_lens, src_masks = self.pad_sequence(dialogues_ids, "dialogues")
            padded_pos_ids, _, _ = self.pad_sequence(pos_ids, "positions")
        
            for counter, role_id in enumerate(role_ids):
            #print(role_id)
                if len(role_id) != 9:
                    role_id = [3, 3, 3, 3, 3, 3, 3, 3, 3]
                    role_ids[counter] = role_id

            data = dict()
            data['dialogues'] = dialogues
            data['labels'] = labels
            data['dialogues_ids'] = padded_dialogues
            data['pos_ids'] = padded_pos_ids
            data['dialogues_lens'] = torch.tensor(dialogues_lens).long()
            data['src_masks'] = src_masks
        #print(role_ids)

            data['role_ids'] = torch.tensor(role_ids).long()
            data['labels_ids'] = torch.tensor(labels_ids).long()
            return data
        except ValueError:
            data = dict()
            return data

    def pad_sequence(self, seqs, what=''):
        #print("Padding seqs {}".format(what))
        #print(seqs)
        lens = [len(seq) for seq in seqs]
        #print("lens")
        #print(lens)
        max_seq_length = max(lens)
        #print("max seq length")
        #print(max_seq_length)
        
        padded_seqs = torch.zeros(len(seqs), max_seq_length).long()
        
        for i, seq in enumerate(seqs):
            end_idx = lens[i]
            padded_seqs[i, :end_idx] = torch.LongTensor(seq[:end_idx])

        src_masks = _gen_seq_bias_mask(lens, max_seq_length)

        return padded_seqs, lens, src_masks

    def tokenize(self, sentence):
        return sentence.split()

    def build_counter(self):
        counter = Counter()
        role_counter = Counter()
        pos_counter = Counter()

        role_words = []
        sentence_words = []
        pos_words = []

        augmented_data_list = [] # add dev vocab
        augmented_data_list.extend(self.data_list)

        dev_examples = torch.load(self.hparams.data_dir + 'dev_corpus')

        print('[%s] %d examples is loaded' % ('Dev', len(dev_examples)))
        for key, value in dev_examples.items():
            texts = value['texts']
            labels = value['labels']
            dialogues = []
            for each in texts:
                role = each[1]
                sentence = ' '.join(word_pos.split('/')[0] for word_pos in each[2].split())
                sentence = sentence.strip().lower()
                pos_sentence = ' '.join(word_pos.split('/')[1] for word_pos in each[2].split())
                pos_sentence = pos_sentence.strip().lower()
                dialogues.append({'role': role, 'sentence': sentence, 'pos_sentence': pos_sentence})
            augmented_data_list.append({'labels': labels, 'dialogues': dialogues})

        for data in augmented_data_list:
            dialogues = data['dialogues']
            for dialogue in dialogues:
                sentence_words.append(self.tokenize(dialogue['sentence']))
                pos_words.append(self.tokenize(dialogue['pos_sentence']))
                role_words.append([dialogue['role']])
            labels = data['labels']
            sentence_words.append(self.tokenize(labels))

        for words in sentence_words:
            counter.update(words)

        for role in role_words:
            role_counter.update(role)

        for pos in pos_words:
            pos_counter.update(pos)
        #print('role_counter: ', role_counter)
        #print('pos_counter: ', pos_counter)

        return counter, role_counter, pos_counter

    def build_vocab(self, counter, max_vocab_size, type='word'):
        vocab = AttrDict()
        if type == 'word':
            print("\n===== Building [Word Vocab] =========")
        elif type == 'role':
            print("\n===== Building [Role Vocab] =========")
        elif type == 'pos':
            print("\n===== Building [POS Vocab] =========")

        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS,
                          '<UNK>': UNK, '<BEGIN>': BEGIN, '<END>': END}
        preset_vocab_size = len(vocab.token2id)
        print('preset_vocab_size: ', preset_vocab_size)
        vocab.token2id.update(
            {token: _id + preset_vocab_size for _id, (token, count) in
             tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        print('Vocab size: ', len(vocab.token2id))
        print('==========================================')
        return vocab

    def tokens2ids(self, tokens, token2id, is_reference=False, is_role=False):
        seq = []
        if is_reference is False and is_role is False:
            # For input dialogues.
            seq.append(BOS)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(EOS)
        elif is_reference:
            # For target summaries.
            seq.append(BEGIN)
            seq.extend([token2id.get(token, UNK) for token in tokens])
            seq.append(END)

        if is_role:
            seq.extend([token2id.get(token, UNK) for token in tokens])
        return seq