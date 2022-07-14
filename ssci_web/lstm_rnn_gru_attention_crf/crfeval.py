import numpy as np
def get_chunks(seq):
    """
        tag2label = {"O": 0, "B": 1, "M": 2, "E": 3,"S":4}
    """
    chunks = []
    chunk_start = None
    for i, tok in enumerate(seq):
        if tok == 'B':
            chunk_start =i
        elif tok == 'E':
            chunk=(chunk_start,i)
            chunks.append(chunk)
        elif tok == 'S':
            chunk = (i, i)
            chunks.append(chunk)
        else:
            pass
    return chunks
def evaluate(path):
    fr=open(path, encoding='utf-8')
    sentences=fr.read().strip().split('\n\n')
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    for sentence in sentences:
        lab=[]
        lab_pred=[]
        for item in sentence.split('\n'):
            lab.append(item.split('\t')[-2])
            lab_pred.append(item.split('\t')[-1])

        accs += [a == b for (a, b) in zip(lab, lab_pred)]
        lab_chunks = set(get_chunks(lab))
        lab_pred_chunks = set(get_chunks(lab_pred))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    print(p,r,f1,acc)

if __name__ == '__main__':
    for i in range(10):
        evaluate('H:\CRF++-0.58/corpus1/'+str(i)+'/result.txt')
