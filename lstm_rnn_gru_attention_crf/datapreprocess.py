import os
import yaml
from collections import defaultdict
import pickle
import random
import numpy as np


#1.生成特征词典
def vocab_build(config, output_path):
    voc_dir = os.path.join(output_path, "voc")
    if not os.path.exists(voc_dir):
        os.makedirs(voc_dir)
    #特征数
    feature_num=config['model_params']['feature_nums']
    feature_feq=[]

    for i in range(feature_num):
        feature_feq.append(defaultdict(int))

    with open(config['data_params']['path_train'],encoding='utf-8') as fr:
        lines = fr.readlines()

    #统计特征频次
    for line in lines:
        line = line.rstrip()
        if not line:#if line == '':
            continue
        items=line.split('\t')
        for i in range(feature_num):
            feature_feq[i][items[i]]+=1

    #剔除少于min_count的特征，并dump。
    for i in range(feature_num):
        voc_name = config['data_params']['feature_params'][i]['voc_name']
        voc_path = os.path.join(voc_dir, voc_name)
        feature2id = {}
        min_count = config['data_params']['feature_params'][i]['min_count']
        for key, value in feature_feq[i].items():
            if value>=min_count:
                feature2id[key]=len(feature2id)
        feature2id['<UNK>'] = len(feature2id)
        with open(voc_path, 'wb') as fw:
            pickle.dump(feature2id, fw)


#2.加载预训练词向量或生成随机初始化词向量
def embedding_build(config, output_path):
    '''
    :param config: 配置文件
    :return: fea2id_list 特征字典列表
             feature_embedding_list  特征embed列表
    '''
    # 特征数
    embed_dir = os.path.join(output_path, "voc")
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    feature_num = config['model_params']['feature_nums']
    fea2id_list = []
    feature_embedding_list = []
    for i in range(feature_num):
        embed_name = config['data_params']['feature_params'][i]['embed_name']
        embed_path = os.path.join(embed_dir, embed_name)
        pre_train = config['model_params']['embed_params'][i]['pre_train']
        if pre_train:
            vec_path = config['model_params']['embed_params'][i]['vec_path']
            embed, fea2id = load_emb(vec_path)
            voc_dir = os.path.join(output_path, "voc")
            voc_name = config['data_params']['feature_params'][i]['voc_name']
            voc_path = os.path.join(voc_dir, voc_name)
            with open(voc_path, 'wb') as fw:
                pickle.dump(fea2id, fw)
        else:
            voc_name = config['data_params']['feature_params'][i]['voc_name']
            voc_path = os.path.join(embed_dir, voc_name)
            fea2id = read_dictionary(voc_path)
            embed_dim = config['model_params']['embed_params'][i]['dimension']
            embed = random_embedding(fea2id, embed_dim)
        with open(os.path.join(embed_dir, embed_name), 'wb') as fw:
            pickle.dump(embed, fw)
        fea2id_list.append(fea2id)
        feature_embedding_list.append(embed)
    return fea2id_list, feature_embedding_list



def load_emb(filename, sep=' '):
    '''
    load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: embedding,word2id
    '''
    result = []
    word2id = {}
    f = open(filename, 'r',encoding='utf-8')
    line = f.readline()
    # print line
    word_dim = int(line.split()[1])

    for l in f:
        l = l.strip()
        if l == '':
            continue
        sp = l.split()

        vals = [np.float32(sp[i]) for i in range(1, len(sp))]
        result.append(vals)
        word2id[sp[0]] =len(word2id)
    word2id['<UNK>']= len(word2id)
    result.append(list(np.random.uniform(-0.25, 0.25, word_dim)))
    return np.array(result),word2id
def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat
def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """

    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    return word2id

def embedding_load(config, restore_dir):
    '''
    for test
    :param config: 配置文件
    :return: fea2id_list 特征字典列表
             feature_embedding_list  特征embed列表
    '''
    # 特征数
    feature_num = config['model_params']['feature_nums']
    fea2id_list = []
    feature_embedding_list = []
    restore_dir = os.path.join(restore_dir, "voc")
    for i in range(feature_num):
        voc_name = config['data_params']['feature_params'][i]['voc_name']
        embed_name = config['data_params']['feature_params'][i]['embed_name']
        voc_path = os.path.join(restore_dir, voc_name)
        embed_path = os.path.join(restore_dir, embed_name)

        fea2id = read_dictionary(voc_path)

        embed = read_dictionary(embed_path)

        fea2id_list.append(fea2id)
        feature_embedding_list.append(embed)
    return fea2id_list,feature_embedding_list


#3.数据初始化
def init_data(feature_num, sentences,fea2id_list,label2id):
    data = []

    for sentence in sentences:
        sent=[]
        label=[]
        [sent.append([]) for _ in range(feature_num)]

        items = sentence.split('\n')
        for item in items:
            feature_tokens = item.split('\t')
            if(len(feature_tokens)<feature_num):
                continue
            for i in range(feature_num):
                if(feature_tokens[i] in fea2id_list[i]):
                    #print(fea2id_list[i][feature_tokens[i]])
                    sent[i].append(fea2id_list[i][feature_tokens[i]])
                else:
                    sent[i].append(fea2id_list[i]['<UNK>'])

            label.append(label2id[feature_tokens[-1]])

        data.append((sent, label))
    return data

def get_train_test_data(data,fold):
    '''
    return testdata,traindata
    '''
    random.shuffle(data)

    return data[:len(data)//fold],data[len(data)//fold:]

def batch_yield(data, batch_size, shuffle=True):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    sentences, labels = [], []
    for (sentence, label) in data:


        if len(sentences) == batch_size:
            yield sentences, labels
            sentences, labels = [], []

        sentences.append(sentence)
        labels.append(label)

    if len(sentences) != 0:
        yield sentences, labels

def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    #print(max_len)
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list