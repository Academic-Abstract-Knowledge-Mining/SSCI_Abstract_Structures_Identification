import random
fr=open('CORPUS.txt', encoding='utf-8')
sentences=fr.read().strip().split('\n\n')
fr.close()
random.shuffle(sentences)
num=len(sentences)
folder =10
trash=[]
for i in range(folder):
    chosen=set()
    while len(chosen)<num//folder:
        id=random.randint(0, num-1)
        while id in trash:
            id = random.randint(0, num-1)
        chosen.add(id)
    trash.extend(chosen)
    fw = open(str(i)+'.txt','w', encoding='utf-8')
    for id in chosen:
        fw.write(sentences[id])
        fw.write('\n\n')
    fw.close()
    fw = open(str(i) + '_rest.txt', 'w', encoding='utf-8')
    for id in range(num):
        if id not in chosen:
            fw.write(sentences[id])
            fw.write('\n\n')
    fw.close()