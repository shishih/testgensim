# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import train_batch_sg
import numpy
from collections import Counter
# from gensim.models.word2vec_inner import train_batch_sg

def te():
    documents=[u"今天 天气 真是 好 啊",u"明天 就要 下雨 了，伐 开心"]
    texts=[[token for token in text] for text in documents]
    # print texts
    dictionary = corpora.Dictionary(texts)
    dictionary.save('./tmp/tedic.dict')
    # print dictionary.token2id
    corpus=[[(1,0.5),(2,1.3)],[]]
    # corpora.MmCorpus.serialize('./tmp/tecor.mm',corpus)
    corpus = corpora.MmCorpus('./tmp/tecor.mm')
    
    tfidf=models.TfidfModel(corpus)
    doc_bow = [(0, 1), (1, 1)]
    print tfidf[doc_bow]
def teword():
    # model=Word2Vec.load_word2vec_format('vectorseg.bin',binary=False)
    # sim=model.most_similar(positive=[u'好',u'开心'],negative=[u'下雨'],topn=2)
    # print sim
    documents=[u"今天 天气 真是 好 啊",u"明天 就要 下雨 了，伐 开心"]
    model=Word2Vec(documents,size=20,window=5,min_count=1)
    sim=model.most_similar(positive=[u"好"],topn=2)
    # model.save('./tmp/tevec')
    print sim

    model=Word2Vec.load_word2vec_format('vectorseg.bin',binary=False)
    Word2Vec.intersect_word2vec_format(model,'fieldvec.bin',binary=False)
    Word2Vec.train_batch_sg(model, sentences, alpha, work=None)

def setwordwindow(windowsize):
    tmpstrp=' 1'*windowsize
    tmpstrn=' -1'*windowsize
    
    files=['corpus/positive','corpus/negative']
    with open('corpus/initindex'+str(windowsize),'w+') as fwrite:
        fwrite.write('18164 '+str(windowsize)+'\n')      
        with open(files[0]) as fp:
            datap=fp.readlines()
            print datap
        with open(files[1]) as fp:
            datan=fp.readlines()
        data=[]
        data.extend(datap)
        data.extend(datan)
   
        counter=Counter(data)

        for i in counter:
            if (counter[i] == 1):
                fwrite.write(i[:-1])
                if i in datap:
                    fwrite.write(tmpstrp)
                else:
                    fwrite.write(tmpstrn)
                fwrite.write('\n')


def intersect(windowsize): 
    # merged OK!   
    windowsize=40
    # model=Word2Vec(size=windowsize,min_count=2,sg=1)

    # sentences=LineSentence('corpus/precorpus')
    # model.build_vocab(sentences)
    # model.train(sentences)
    # print 'finish pre-train'
    # model.save('corpus/pretrain'+str(windowsize)+'.model')
    # model.save_word2vec_format('corpus/pretrain'+str(windowsize))
    
    # intersect does not delete the bibary tree, but load does
    model=Word2Vec.load('corpus/pretrain'+str(windowsize)+'.model')
    setwordwindow(windowsize)
    Word2Vec.intersect_word2vec_format(model,'corpus/initindex'+str(windowsize),binary=False)
    print 'finish intersect'
    model.save('corpus/merged'+str(windowsize)+'.model')
    model.save_word2vec_format('corpus/merged'+str(windowsize), binary=False)
    # model=Word2Vec.load_word2vec_format('temerged',binary=False) 

    # model.build_vocab(sentences)

    sensum=0
    for i in sentences:
        sensum+=1
    model.corpus_count=sensum

    # Word2Vec.reset_weights(model)
    sentences=LineSentence('corpus/fieldcorpus')
    
    model.iter=1
    model.train(sentences)

    # in class Word2Vec
    # self.build_vocab(sentences, trim_rule=trim_rule)
    # self.train(sentences)
    #

    # train_batch_sg(model, sentences, alpha=0.1,work=None)
    # simply use train and set iter=1?
    model.save('mergedtrained'+str(windowsize)+'iter'+str(model.iter)+'.model')
    model.save_word2vec_format('mergedtrained'+str(windowsize)+'iter'+str(model.iter), binary=False)


def main():
    # te()
    # teword()
    intersect(40)
    # setwordwindow(40)
    # Word2Vec.load_word2vec_format('initindex.txt',binary=False)
    

if __name__ == '__main__':
    main()