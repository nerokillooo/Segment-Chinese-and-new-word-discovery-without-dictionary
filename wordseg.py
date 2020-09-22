# coding=utf-8

"""
Chinese word segmentation algorithm without corpus
Author: 段凯强
Reference: http://www.matrix67.com/blog/archives/5044
"""

import re
import time
import numpy as np

from probability import entropyOfList
from sequence import genSubparts, genSubstr
from fmm import fmm, particleWord
from filter_data import filter_data
from collections import defaultdict


def indexOfSortedSuffix(doc, max_word_len):
    """
    Treat a suffix as an index where the suffix begins.
    Then sort these indexes by the suffixes.
    """
    indexes = []
    length = len(doc)

    for i in range(0, length):
        for j in range(i + 1, min(i + 1 + max_word_len, length+1)):#########?length+1
            indexes.append((i, j))
    return sorted(indexes, key=lambda i_j: doc[i_j[0]:i_j[1]])

class WordInfo(object):
    """
    Store information of each word, including its freqency, left neighbors and right neighbors
    """

    def __init__(self, text):
        super(WordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []
        self.right = []
        self.aggregation = 0
        self.PMI = 1
        self.logit = 1

    def update(self, left, right):
        """
        Increase frequency of this word, then append left/right neighbors
        @param left a single character on the left side of this word
        @param right as left is, but on the right side
        """
        self.freq += 1
        if left:
            self.left.append(left)
            # print(left)

        if right:
            self.right.append(right)
            # print(right)

    def compute(self, length):
        """
        Compute frequency and entropy of this word
        @param length length of the document for training to get words
        """
        self.freq /= length
        self.left = entropyOfList(self.left)
        #print(self.left)
        self.right = entropyOfList(self.right)

    def computeAggregation(self, words_dict):
        """
        Compute aggregation of this word
        @param words_dict frequency dict of all candidate words
        """
        parts = genSubparts(self.text)
        if len(parts) > 0:
            self.aggregation = min(
                [self.freq / words_dict[p1_p2[0]].freq / words_dict[p1_p2[1]].freq for p1_p2 in parts])
            self.PMI = np.log(self.aggregation)
            self.logit = np.log(self.freq)



class WordSegment(object):
    """
    Main class for Chinese word segmentation
    1. Generate words from a long enough document
    2. Do the segmentation work with the document
    """

    # if a word is combination of other shorter words, then treat it as a long word
    L = 0
    # if a word is combination of other shorter words, then treat it as the set of shortest words
    S = 1
    # if a word contains other shorter words, then return all possible results
    ALL = 2

    def __init__(self, doc, max_word_len=5, min_freq=0.00005, min_entropy=2.0, min_aggregation=50):
        super(WordSegment, self).__init__()
        self.word_cands = {}
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_aggregation = min_aggregation
        self.word_infos = self.genWords(doc)
        # Result infomations, i.e., average data of all words
        word_count = float(len(self.word_infos))
        # print(word_count)
        self.avg_len = sum([len(w.text) for w in self.word_infos]) / word_count
        self.avg_freq = sum([w.freq for w in self.word_infos]) / word_count
        self.avg_left_entropy = sum([w.left for w in self.word_infos]) / word_count
        self.avg_right_entropy = sum([w.right for w in self.word_infos]) / word_count
        self.avg_aggregation = sum([w.aggregation for w in self.word_infos]) / word_count
        # Filter out the results satisfy all the requirements
        filter_func = lambda v: len(v.text) > 1 and v.aggregation > self.min_aggregation and \
                                v.freq > self.min_freq and v.left > self.min_entropy and v.right > self.min_entropy
        self.word_with_entropy = [(w.text, w.left, w.right) for w in list(filter(filter_func, self.word_infos))]
        self.word_with_aggregation = [(w.text, w.aggregation) for w in list(filter(filter_func, self.word_infos))]
        self.words = [w[0] for w in self.word_with_entropy]
        #self.new_words = dict(filter(lambda key: key in self.words, self.word_cands.keys()))
        self.new_word_cands = {}




    def genWords(self, doc):
        """
        Generate all candidate words with their frequency/entropy/aggregation informations
        @param doc the document used for words generation
        """
        # [\\s\\d,.·<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]
        length = len(doc)

        count = 1
        pattern = r'\n|[a-zA-Z_0-9]|\s|\n|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）|“|”|：|——|？|%|《|》'
        #pattern = r'。'
        #print(length)
        for sentence in filter_data(re.split(pattern, doc)):
            #print(count, sentence)
            count+=1
            if sentence:
                # doc = re.sub(pattern, '', str(doc))
                suffix_indexes = indexOfSortedSuffix(sentence, self.max_word_len)
                #print(suffix_indexes)

                # compute frequency and neighbors
                for suf in suffix_indexes:
                    word = sentence[suf[0]:suf[1]]
                    # print(word)
                    if word not in self.word_cands:
                        self.word_cands[word] = WordInfo(word)
                    self.word_cands[word].update(sentence[suf[0] - 1:suf[0]], sentence[suf[1]:suf[1] + 1])

                # compute probability and entropy
                #length = len(doc)
        for k in self.word_cands:
            self.word_cands[k].compute(length)
        #print('aaaa')
        #print('value', word_cands.values())

        # compute aggregation of words whose length > 1
        values = sorted(list(self.word_cands.values()), key=lambda x: len(x.text), reverse=True)
        #print('values', values)
        for v in values:
            if len(v.text) == 1: continue
            v.computeAggregation(self.word_cands)

        return sorted(values, key=lambda v: len(v.text), reverse=True)

    def segSentence(self, sentence, method=ALL):
        """
        Segment a sentence with the words generated from a document
        @param sentence the sentence to be handled
        @param method segmentation method
        """
        i = 0
        res = []
        while i < len(sentence):
            if method == self.L or method == self.S:
                j_range = list(range(self.max_word_len, 0, -1)) if method == self.L else list(
                    range(2, self.max_word_len + 1)) + [1]
                for j in j_range:
                    if j == 1 or sentence[i:i + j] in self.words:
                        res.append(sentence[i:i + j])
                        i += j
                        break
            else:
                to_inc = 1
                for j in range(2, self.max_word_len + 1):
                    if i + j <= len(sentence) and sentence[i:i + j] in self.words:
                        res.append(sentence[i:i + j])
                        if to_inc == 1: to_inc = j
                if to_inc == 1: res.append(sentence[i])
                i += to_inc
        # print(res)
        return res












if __name__ == '__main__':
    start = time.process_time()

    # doc = '十四是十四四十是四十，，十四不是四十，，，，四十不是十四'

    pathNew = 'C:\\Users\\N\\Desktop\\wordseg\\news_0.txt'
    pathDict = 'C:\\Users\\N\\Desktop\\wordseg\\dictionary.txt'
    fileNew = open(pathNew, encoding='utf-8', errors='ignore')
    fileDict = open(pathDict, encoding='utf-8', errors='ignore')


    doc = fileNew.read()
    dict = fileDict.read()
    wsNew = WordSegment(doc, max_word_len=5, min_freq=0.00017, min_aggregation=120, min_entropy=0.693)
    #wsNew.find_sub(wsNew.word_cands)
    #ws = wsNew.find(wsNew.word_cands)


    ws = sorted(wsNew.words, key=lambda i: len(i), reverse=True)
    print(len(ws))
    print('ws', ws)
    # Reduc.write(''.join(ws))
    # print('reduc',Reduc)
    '''
    doc2 = ws
    print('doc2',doc2)
    for w in doc2:
         newdoc = doc2
         newdoc.remove(w)
         wsRes = fmm(w, newdoc)'''

    #print('wsRes.words',wsRes)

    ns = []
    count = 0
    for w in ws:
        if dict.find(w) == -1:
            wfs = fmm(w, dict)
            #wp = particleWord(wfs, dict)
            #if wp: continue
            count += 1
            ns.append(w)
    end = time.process_time()
    print(count)
    print('ns', ns)

    output = open('C:\\Users\\N\\Desktop\\wordseg\\output.txt', 'w')
    output.write('\n'.join(ns))
    # print(count)
    #print('\n'.join(ns))

    print(' '.join(['%s:left %f right %f' % w for w in wsNew.word_with_entropy if w[0] in ns]))
    print(' '.join(['%s: agg %f' % w for w in wsNew.word_with_aggregation if w[0] in ns]))
    print('average left entropy: ', wsNew.avg_left_entropy)
    print('average right entropy: ', wsNew.avg_right_entropy)
    print('average aggregation: ', wsNew.avg_aggregation)

    print('Running time: %s Seconds' % (end - start))  # [0.7656525, 0.875]->updated: 0.6718-> updated：0.578125
    '''
    print(' '.join(wsDict.words))
    print(' '.join(wsNew.segSentence(ns)))
    print('average len: ', wsDict.avg_len)
    print('average frequency: ', wsDict.avg_freq)
    
    print('average aggregation: ', wsDict.avg_aggregation)
    '''
