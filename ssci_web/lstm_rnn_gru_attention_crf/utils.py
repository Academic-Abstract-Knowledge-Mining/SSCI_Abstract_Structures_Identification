import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG
def get_chunks(seq, tags):
    """
        tag2label = {"O": 0, "B": 1, "M": 2, "E": 3,"S":4}
    """
    chunks = []
    chunk_start = None
    for i, tok in enumerate(seq):
        if tok == 1:
            chunk_start =i
        elif tok == 3:
            chunk=(chunk_start,i)
            chunks.append(chunk)
        elif tok == 4:
            chunk = (i, i)
            chunks.append(chunk)
        else:
            pass
    return chunks
def get_NT_entity(tag_seq, char_seq):
    length = len(char_seq)
    NT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-nt':
            if 'nt' in locals().keys():
                NT.append(nt)#下一个实体开始，输出
                del nt
            nt = char#词头：初始一个per
            if i+1 == length:#到序列末尾，直接输出
                NT.append(nt)
        if tag == 'I-nt':
            nt += char#词中：字符串append
            if i+1 == length:
                NT.append(nt)
        if tag == 'E-nt':
            nt += char#词尾：字符串append
            NT.append(nt)  #输出
            del nt
        if tag == 'S-nt':#tag e,s,
            if 'nt' in locals().keys():
                NT.append(nt)
                del nt
            NT.append(char)
        if tag not in ['I-nt', 'B-nt','E-nt','S-nt']:#tag e,s,
            if 'nt' in locals().keys():
                NT.append(nt)
                del nt
            continue

    return NT
def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)#下一个实体开始，输出
                del per
            per = char#词头：初始一个per
            if i+1 == length:#到序列末尾，直接输出
                PER.append(per)
        if tag == 'I-PER':
            per += char#词中：字符串append
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:#tag e,s,o
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG

