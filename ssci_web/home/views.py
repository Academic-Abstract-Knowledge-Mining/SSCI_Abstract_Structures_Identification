import base64

from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import yaml
from lstm_rnn_gru_attention_crf.api import load_model, model_predict
import json
from elasticsearch import Elasticsearch
from model.api import StructPredictModel

es = Elasticsearch()
cls_model = StructPredictModel(model_dir=r"D:\python\ssci_web\model\save_m\checkpoints")
cls_model.load_parameter()

with open('lstm_rnn_gru_attention_crf/config.yml', encoding='utf-8') as file_config:
    config = yaml.load(file_config)

seq_model = load_model(config)

def home(request):
    return render(request, template_name="index.html")


def struct(request):
    return render(request, template_name="struct.html")


def seq_model_predict(text):
    feature_num = seq_model.feature_num
    fea2id_list = seq_model.fea2id_list
    label2id = seq_model.label2id
    sentences = [
        text.split()
    ]
    data = []
    for s in sentences:
        sent = []
        label = []
        [sent.append([]) for _ in range(feature_num)]
        for word in s:
            for i in range(feature_num):
                if(word in fea2id_list[i]):
                    sent[i].append(fea2id_list[i][word])
                else:
                    sent[i].append(fea2id_list[i]['<UNK>'])
            label.append(label2id['B-B'])
        data.append((sent, label))
    label = model_predict(data, seq_model)
    result = {
        "text": sentences[0],
        "label": label[0],
    }
    return result

def cls_model_predict(text):
    result = {}
    sentence = [s for s in text.split(".") if s]
    predict = cls_model.predict(sentence)
    for i in range(len(predict)):
        result.setdefault(predict[i], [])
        result[predict[i]].append(sentence[i])
    return result


def get_struct(request):
    text = request.GET.get("text").strip()
    method = request.GET.get("method").strip()
    if method == "cls":
        result = cls_model_predict(text)
    else:
        result = seq_model_predict(text)
    return HttpResponse(json.dumps(result), content_type="application/json;charset = utf-8")


def detail(request):
    id = request.GET.get("id", None)
    doc = es.get(id=int(id), index="wos_struct_idx")
    abstract = ""
    if doc["_source"]["Background"]:
        abstract += "<strong>Background: </strong>" + doc["_source"]["Background"]
    if doc["_source"]["Purpose"]:
        abstract += "<strong>Purpose: </strong>" + doc["_source"]["Purpose"]
    if doc["_source"]["Method"]:
        abstract += "<strong>Method: </strong>" + doc["_source"]["Method"]
    if doc["_source"]["Result"]:
        abstract += "<strong>Result: </strong>" + doc["_source"]["Result"]
    if doc["_source"]["Conclusion"]:
        abstract += "<strong>Conclusion: </strong>" + doc["_source"]["Conclusion"]
    doc["_source"]["abstract"] = abstract
    return render(request, template_name="detail.html", context=doc["_source"], )


def about(request):
    return render(request, template_name="about.html")


def search(request):
    key_word = request.GET.get('key_word', None)
    offset = int(request.GET.get('offset', None))
    limit = int(request.GET.get('limit', None))
    field = request.GET.get('field', None)
    dsl = {
        'from': offset,
        'size': limit,
        'query': {
            'match': {
                field: key_word
            }
        },
        "highlight": {
            "fields": {
                field: {},
            }
        }
    }
    if key_word:
        res = es.search(index="wos_struct_idx", body=dsl)
        rows = []
        for doc_rank, r in enumerate(res['hits']['hits']):
            item = r['_source']
            item['abstract'] = " ".join(r['highlight'][field])
            item['doc_rank'] = doc_rank
            item["year"] = r["_source"]["PY"]
            item["au"] = r["_source"]["AU"]
            item["title"] = r["_source"]["TI"]
            item["id"] = r["_id"]
            rows.append(item)
        result = {
            'rows': rows,
            'total': res['hits']['total']['value']
        }
    else:
        result = {
            'rows': [],
            'total': 0,
        }
    return HttpResponse(json.dumps(result), content_type="application/json;charset = utf-8")