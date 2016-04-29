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
    tmpstrp=' 1'*windowsize
    tmpstrn=' -1'*windowsize
    
    files=['corpus/positive','corpus/negative']
    with open('corpus/initindex'+str(windowsize),'w+') as fwrite:       
        with open(files[0]) as fp:
            for col in fp:
                fstr=col[:-1]+tmpstrp+'\n'
                fwrite.write(fstr)
        with open(files[1]) as fp:
            for col in fp:
                fstr=col[:-1]+tmpstrn+'\n'
                fwrite.write(fstr)


def intersect(windowsize): 
    # merged OK!   
    # syn0 !
    windowsize=40
    model=Word2Vec(size=windowsize,min_count=2,sg=1)

    sentences=LineSentence('corpus/precorpus')
    model.build_vocab(sentences)
    model.train(sentences)
    print 'finish pre-train'
    # print model.syn0,model.syn0_lockf
    
    # intersect does not delete the bibary tree, but load does
    setwordwindow(windowsize)
    Word2Vec.intersect_word2vec_format(model,'corpus/initindex'+str(windowsize),binary=False)
    print 'finish intersect'
    model.save_word2vec_format('corpus/merged'=str(windowsize), binary=False)
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
    sentences=LineSentence('corpus/fieldcorpus')
    
    model.iter=1
    model.train(sentences)

    # in class Word2Vec
    # self.build_vocab(sentences, trim_rule=trim_rule)
    # self.train(sentences)
    #

    # train_batch_sg(model, sentences, alpha=0.1,work=None)
    # simply use train and set iter=1?
    model.save('mergedtrained'+str(windowsize)+'.model')
    model.save_word2vec_format('mergedtrained'+str(windowsize), binary=False)


def main():
    # te()
    # teword()
    intersect(40)
    # setwordwindow(40)
    

if __name__ == '__main__':
    main()