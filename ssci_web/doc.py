import json
from elasticsearch import Elasticsearch

es = Elasticsearch()

def main():
    with open("info.json", "r") as fr:
        docs = json.load(fr)

    for d in docs:
        if d["AB"]:
            is_struct = d.pop("struct")
            wos_id = d.pop("WOS_ID")
            if is_struct:
                ab_stuct = d.pop("ab_struct")
                print(ab_stuct)
                # for s in ab_stuct:
                #     d[s] = ab_stuct[s]
                # es.index(index="wos_struct_idx", id=wos_id, body=d)


if __name__ == '__main__':
    main()


