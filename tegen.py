# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import train_batch_sg
import numpy as np
from collections import Counter
import re
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier


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

def setwordwindow(vectorsize):
    tmpstrp=' 1'*vectorsize
    tmpstrn=' -1'*vectorsize
    pattern=re.compile(r' ')
    
    files=['corpus/positive','corpus/negative']
    with open('corpus/initindex'+str(vectorsize),'w+') as fwrite:
        fwrite.write('15289 '+str(vectorsize)+'\n')      
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
                i=re.sub(r'\s+','',i)
                fwrite.write(i)
                if i in datap:
                    fwrite.write(tmpstrp)
                else:
                    fwrite.write(tmpstrn)
                fwrite.write('\n')


def intersect(vectorsize): 
    # merged OK!   
    # vectorsize=40
    model=Word2Vec(size=vectorsize,min_count=2,sg=1)

    sentences=LineSentence('corpus/precorpus')
    model.build_vocab(sentences)
    model.train(sentences)
    print 'finish pre-train'
    model.save('corpus/pretrain'+str(vectorsize)+'.model')
    model.save_word2vec_format('corpus/pretrain'+str(vectorsize))
    
    # intersect does not delete the bibary tree, but load does
    # model=Word2Vec.load('corpus/pretrain'+str(vectorsize)+'.model')
    setwordwindow(vectorsize)
    Word2Vec.intersect_word2vec_format(model,'corpus/initindex'+str(vectorsize),binary=False)
    print 'finish intersect'
    model.save('corpus/merged'+str(vectorsize)+'.model')
    model.save_word2vec_format('corpus/merged'+str(vectorsize), binary=False)


    # model.build_vocab(sentences)

    # sensum=0
    # for i in sentences:
    #     sensum+=1
    # model.corpus_count=sensum

    # Word2Vec.reset_weights(model)
    # model=Word2Vec.load('corpus/merged40.model')
    print "finish load"
    sentences=LineSentence('corpus/fieldcorpus')
    print "finish sentence building"
    
    model.iter=1
    model.train(sentences)
    print "finish training"

    # in class Word2Vec
    # self.build_vocab(sentences, trim_rule=trim_rule)
    # self.train(sentences)
    #

    # train_batch_sg(model, sentences, alpha=0.1,work=None)
    # simply use train and set iter=1?
    model.save('corpus/mergedtrained'+str(vectorsize)+'iter'+str(model.iter)+'.model')
    model.save_word2vec_format('corpus/mergedtrained'+str(vectorsize)+'iter'+str(model.iter), binary=False)

def dis(model):
    # print model.similarity("今天","在")
    print model.similarity(u"分手",u"好")
    print model.similarity(u"分手",u"坏")

def buildvector(model,x,vectorsize):
    # size=x.length
    vec=np.zeros(vectorsize)
    count=0
    for i in x:
        try:
            vec+=model[x]
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count 
    return vec

def classify(model,vectorsize):
    with open('corpus/testclass.txt') as fp:
        # print fp.readlines()[0]
        xl=[]
        yl=[]
        for i in fp:
            yl.append(int(i[0]))
            xl.append(i[2:-1].split(' ')) 
    y=np.array(yl)
    x=np.array(xl)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    # print y_train
    vec_train=np.array([buildvector(model,x,vectorsize) for x in x_train])
    vec_test=np.array([buildvector(model,x,vectorsize) for x in x_test])

    lr=SGDClassifier(loss='log')
    lr.fit(vec_train,y_train)
    print lr.score(vec_test,y_test)


def main():
    # te()
    # teword()
    # intersect(40)
    # setwordwindow(40)
    # Word2Vec.load_word2vec_format('corpus/initindex40',binary=False)
    model=Word2Vec.load('corpus/mergedtrained40iter1.model')

    dis(model)
    # classify(model,40)
    

if __name__ == '__main__':
    main()