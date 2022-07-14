import os
from lstm_rnn_gru_attention_crf.datapreprocess import embedding_load, init_data
from lstm_rnn_gru_attention_crf.model import MF_SequenceLabelingModel


def load_model(config):
    config['data_params']['label2id'] = \
        dict(zip(config['data_params']['label2id'], range(len(config['data_params']['label2id']))))
    print('配置文件加载成功')
    model_restore_path = os.path.join('.', config['model_params']['model_restore_path'])
    # 2.加载词向量
    print('加载embedding')
    fea2id_list, feature_embedding_list = embedding_load(config, model_restore_path)
    feature_num = config['model_params']['feature_nums']
    feature_weight_dropout_list = []
    for i in range(feature_num):
        feature_weight_dropout_list.append(config['model_params']['embed_params'][i]['dropout_rate'])

    # 4.加载标签2id
    label2id = config['data_params']['label2id']
    num_class = len(label2id)

    # 5.读取模型参数
    batch_size = config['model_params']['batch_size']
    epoch_num = config['model_params']['epoch_num']
    max_patience = config['model_params']['max_patience']  # early stop

    num_layers = config['model_params']['num_layers']
    rnn_unit = config['model_params']['rnn_unit']  # rnn 类型
    hidden_dim = config['model_params']['hidden_dim']  # rnn 单元数

    dropout = config['model_params']['dropout_rate']
    optimizer = config['model_params']['optimizer']
    lr = config['model_params']['learning_rate']

    clip = config['model_params']['clip']

    use_crf = config['model_params']['use_crf']
    is_attention = config['model_params']['is_attention']
    model = MF_SequenceLabelingModel(feature_embedding_list, feature_num, feature_weight_dropout_list, fea2id_list,
                                     label2id, num_class,
                                     batch_size, epoch_num, max_patience, num_layers, rnn_unit, hidden_dim,
                                     dropout, optimizer, lr, clip, use_crf, model_restore_path, is_attention, config)
    model.restore_para()
    return model


def model_predict(data, model):
    evalformat_data = model.predict(data)
    return evalformat_data