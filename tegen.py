# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import train_batch_sg
import numpy
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
    tmpstr=u' 1'*windowsize
    
    files=['positive.txt','negative.txt']
    for file in files:
        with open(file+str(windowsize),'a') as fwrite:
            with open(file) as fp:
                for col in fp:
                    fwrite.write(col+tmpstr)
                

def teintersect(): 
    # merged OK!   
    # syn0 !
    window=40
    model=Word2Vec(size=40,min_count=2,sg=1)

    sentences=LineSentence('parttotrain')
    model.build_vocab(sentences)
    model.train(sentences)
    print 'finish pre-train'
    # print model.syn0,model.syn0_lockf
    
    # intersect does not delete the bibary tree, but load does
    Word2Vec.intersect_word2vec_format(model,'initindex.txt',binary=False)
    print 'finish intersect'
    model.save_word2vec_format('temerged', binary=False)
    # model=Word2Vec.load_word2vec_format('temerged',binary=False)


    # Word2Vec.make_cum_table(model)
    # print 'model.cum_table',model.cum_table
    # model.syn0_lockf = numpy.ones(len(model.vocab)) 

    # print model.vocab,model.syn0_lockf    

    # model.build_vocab(sentences)

    sensum=0
    for i in sentences:
        sensum+=1
    model.corpus_count=sensum

    # Word2Vec.reset_weights(model)
    sentences=LineSentence('partseganswer')
    
    model.iter=1
    model.train(sentences)

    # in class Word2Vec
    # self.build_vocab(sentences, trim_rule=trim_rule)
    # self.train(sentences)
    #

    # train_batch_sg(model, sentences, alpha=0.1,work=None)
    # simply use train and set iter=1?
    model.save('mergedtrained.model')
    model.save_word2vec_format('temergedtrained'+str(window), binary=False)


def main():
    # te()
    # teword()
    # teintersect()
    

if __name__ == '__main__':
    main()