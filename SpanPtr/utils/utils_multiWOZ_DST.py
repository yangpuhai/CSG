import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
#from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from .fix_label import *

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

def seq2extend_ids(lis, word2idx):
    ids = []
    oovs = []
    oovs_word2idx={}
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(word2idx) + oov_num)
            oovs_word2idx[w]=len(word2idx) + oov_num
    oovs_word2idx.update(word2idx)
    return ids, oovs, oovs_word2idx

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type, value_list):
        if type == 'utter':
            for word in sent.split():
                self.index_word2(word,value_list)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word2(d,value_list)
                for ss in s.split():
                    self.index_word2(ss,value_list)
                for v in value.split():
                    self.index_word2(v,value_list)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def index_word2(self, word, value_list):
        if word not in self.word2index and word not in value_list:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, trg_n_words, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id
        self.trg_n_words = trg_n_words
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        context = self.dialog_history[index]
        ids, oov_list, oovs_word2idx = seq2extend_ids(context.split(), self.trg_word2id)
        oovs_idx2word = {v: k for k, v in oovs_word2idx.items()}
        generate_y = self.generate_y[index]
        #print('/n')
        #print(context.split())
        #print(generate_y)
        #print(len(oovs_word2idx))
        generate_y_in = self.preprocess_slot(generate_y, self.trg_word2id)
        generate_y_out = self.preprocess_slot(generate_y, oovs_word2idx)
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]
        enc_batch_extend_vocab_idx = torch.Tensor(ids)
        #print(generate_y_in)
        #print(generate_y_out)
        extend_vocab_len = self.trg_n_words + len(oov_list)
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief, 
            "gating_label":gating_label, 
            "context":context, 
            "context_plain":context_plain, 
            "turn_uttr_plain":turn_uttr, 
            "turn_domain":turn_domain, 
            "generate_y_in":generate_y_in,
            "generate_y_out": generate_y_out,
            "enc_batch_extend_vocab_idx":enc_batch_extend_vocab_idx,
            "oov_list":oov_list,
            "oovs_idx2word":oovs_idx2word,
            "extend_vocab_len":extend_vocab_len
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths


    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    extend_vocab_src, _ = merge(item_info['enc_batch_extend_vocab_idx'])
    y_in_seqs, y_in_lengths = merge_multi_response(item_info["generate_y_in"])
    y_out_seqs, y_out_lengths = merge_multi_response(item_info["generate_y_out"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    oov_list_len=max([len(li) for li in item_info['oov_list']])
    if oov_list_len == 0:
        extra_zeros = None
    else:
        extra_zeros = torch.zeros((len(item_info['oov_list']), oov_list_len))

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        extend_vocab_src = extend_vocab_src.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_in_seqs = y_in_seqs.cuda()
        y_in_lengths = y_in_lengths.cuda()
        y_out_seqs = y_out_seqs.cuda()
        y_out_lengths = y_out_lengths.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()


    item_info["context"] = src_seqs
    item_info['enc_batch_extend_vocab_idx'] = extend_vocab_src
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y_in"] = y_in_seqs
    item_info["y_in_lengths"] = y_in_lengths
    item_info["generate_y_out"] = y_out_seqs
    item_info["y_out_lengths"] = y_out_lengths
    item_info["extra_zeros"] = extra_zeros
    return item_info

def read_langs(value_list, file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {}
    sum1=0
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first
        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

                if (args["all_vocab"] or dataset=="train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief', value_list)

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])
                
                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "dialog_history":source_text+' EOS',
                    "turn_belief":turn_belief_list,
                    "gating_label":gating_label, 
                    "turn_uttr":turn_uttr_strip, 
                    'generate_y':generate_y
                    }

                if not turn_belief_dict:
                    continue

                if dataset == "train":
                    tag1=0
                    for value in generate_y:
                        if value is not "none" and value is not "dontcare" and value not in source_text:
                            tag1=1
                            break
                    if tag1==1:
                        continue

                if dataset == "dev" or dataset == "test":
                    for k, v in turn_belief_dict.items():
                        for v1 in v.split():
                            if v1 not in value_list:
                                value_list.append(v1)

                else:
                    if (args["all_vocab"] or dataset == "train") and training:
                            lang.index_words(turn_uttr, 'utter', value_list)

                data.append(data_detail)
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())
                
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter', [])

    print("domain_counter", domain_counter)
    return value_list, data, max_resp_len, slot_temp


def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):  
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang.n_words, sequicity, mem_lang.word2index)

    if args["imbalance_sampler"] and type:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type,
                                                  collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    #embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    embeddings = [0, 0]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS

def avoid_word(value_list):
    avoid=['none','dontcare','the','and','by','from','of','at','to']
    for a in avoid:
        if a in value_list:
            value_list.remove(a)
    return value_list

def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    value_oov_rate = args["value_oov_rate"]
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'
    # Create saving folder
    if args['path']:
        #folder_name = args['path'].rsplit('/', 2)[0] + '/'
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        #folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
        folder_name = 'save/{}-'.format(args["decoder"]) + args["addName"] + args['dataset'] + str(args['task']) +'-'+str(args["value_oov_rate"])+'-'+args['mode']+ '/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2.1/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot', [])
    mem_lang.index_words(ALL_SLOTS, 'slot', [])
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'
    value_list = []
    if training:
        value_list, pair_dev, dev_max_len, slot_dev = read_langs(value_list, file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        value_list, pair_test, test_max_len, slot_test = read_langs(value_list, file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        value_list = avoid_word(value_list)
        print('len(value_list):', len(value_list))
        random.seed(0)
        rate = int(len(value_list) * value_oov_rate)
        value_list = random.sample(value_list, rate)
        value_list, pair_train, train_max_len, slot_train = read_langs(value_list, file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        if os.path.exists(folder_name + lang_name) and os.path.exists(folder_name + mem_lang_name):
            os.remove(folder_name + lang_name)
            os.remove(folder_name + mem_lang_name)
        print("[Info] Dumping lang files...")
        with open(folder_name + lang_name, 'wb') as handle:
            pickle.dump(lang, handle)
        with open(folder_name + mem_lang_name, 'wb') as handle:
            pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        value_list, pair_dev, dev_max_len, slot_dev = read_langs(value_list, file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        value_list, pair_test, test_max_len, slot_test = read_langs(value_list, file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if args['except_domain']!="":
        value_list, pair_test_4d, _, _ = read_langs(value_list, file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d  = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples