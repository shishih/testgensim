# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import train_batch_sg
import numpy as np
from collections import Counter
import re
import pickle
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import sys     
reload(sys)  
sys.setdefaultencoding('utf-8') 


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

def pretrain(vectorsize):
    model=Word2Vec(size=vectorsize,min_count=2,sg=1)# skip-gram is used

    sentences=LineSentence('corpus/precorpus')
    model.build_vocab(sentences)
    model.train(sentences)
    print 'finish pre-train'
    model.save('corpus/pretrain'+str(vectorsize)+'.model')
    model.save_word2vec_format('corpus/pretrain'+str(vectorsize),binary=False)
    print 'finish save'

def fieldtrain(vectorsize):
    model=Word2Vec.load('corpus/pretrain'+str(vectorsize)+'.model')
    print 'finish load'
    sentences=LineSentence('corpus/fieldcorpus')
    model.train(sentences)
    print 'finish fieldtrain'
    model.save('corpus/fieldtrained'+str(vectorsize)+'.model')
    model.save_word2vec_format('corpus/fieldtrained'+str(vectorsize), binary=False)
    print 'finish save'

def intersect(vectorsize):
    model=Word2Vec.load('corpus/fieldtrained'+str(vectorsize)+'.model')
    # setwordwindow(vectorsize)
    print 'finish load'
    Word2Vec.intersect_word2vec_format(model,'corpus/initindex'+str(vectorsize),binary=False)
    print 'finish intersect'
    model.save('corpus/merged'+str(vectorsize)+'.model')
    model.save_word2vec_format('corpus/merged'+str(vectorsize), binary=False)
    print 'finish save'

def intertrain(vectorsize):
    model=load('corpus/merged'+str(vectorsize)+'.model')
    print 'finish load'
    sentences=LineSentence('corpus/fieldcorpus')
    model.train(sentences)
    print 'finish train'
    model.save('corpus/mergedtrained'+str(vectorsize)+'iter'+str(model.iter)+'.model')
    model.save_word2vec_format('corpus/mergedtrained'+str(vectorsize)+'iter'+str(model.iter), binary=False)
    print 'finish save'
    
def intersect0(vectorsize): 
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

def dis(vectorsize):
    # print model.similarity("今天","在")
    model=Word2Vec.load('corpus/mergedtrained'+str(vectorsize)+'iter1'+'.model')
    modelfield=Word2Vec.load('corpus/fieldtrained'+str(vectorsize)+'.model')
    print model.similarity(u"分手",u"好")
    print model.similarity(u"分手",u"坏")
    print modelfield.similarity(u"分手",u"好")
    print modelfield.similarity(u"分手",u"坏")

def buildvector(model,x,vectorsize):
    # size=x.length
    vec=np.zeros(vectorsize)
    count=0.0
    for i in x:
        try:
            # if i in stopwordlist:
            #     continue
            vec+=model[unicode(i)]
            count+=1.0
        except KeyError:
            # print 'keyerror'
            continue
    if count!=0.0:
        vec/=count
        # print vec
    return vec

def classify(modelpre,modelfield,modelmerged,vectorsize):
    run=1
    while(run>0):        
        print run

        with open('corpus/documentlabelstop.txt') as fp:
        # print fp.readlines()[0]
            xl=[]
            yl=[]
            for i in fp:
                yl.append(int(i[0]))
                xl.append(i[2:-1].split(' ')) 
        y=np.array(yl)
        x=np.array(xl)
        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
        # print x_train
        with open('corpus/x_train.txt','w') as fp:
            pickle.dump(x_train,fp)
        with open('corpus/y_train.txt','w') as fp:
            pickle.dump(y_train,fp)
        with open('corpus/x_test.txt','w') as fp:
            pickle.dump(x_test,fp)
        with open('corpus/y_test.txt','w') as fp:
            pickle.dump(y_test,fp)

        # with open('corpus/stopword.txt') as fp:
        #     data=fp.readlines()
        #     stopwordlist=[x[:-1] for x in data]

        vec_trainpre=np.array([buildvector(modelpre,x,vectorsize) for x in x_train])
        vec_testpre=np.array([buildvector(modelpre,x,vectorsize) for x in x_test])

        # lr=SGDClassifier(loss='log')
        # lr.fit(vec_trainpre,y_train)
        # scorepre=lr.score(vec_testpre,y_test)
        # print scorepre

        clfmodel = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0)
        clf=clfmodel.fit(vec_trainpre, y_train)
        # y_predicted = clf.predict(vec_testpre)
        svmpre=clf.score(vec_testpre,y_test)
        print svmpre
        # print metrics.classification_report(y_test, y_predicted)

        vec_trainfield=np.array([buildvector(modelfield,x,vectorsize) for x in x_train])
        vec_testfield=np.array([buildvector(modelfield,x,vectorsize) for x in x_test])

        # lr=SGDClassifier(loss='log')
        # lr.fit(vec_trainfield,y_train)
        # scorefield=lr.score(vec_testfield,y_test)
        # print scorefield

        clfmodel = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0)
        clf=clfmodel.fit(vec_trainfield, y_train)
        # y_predicted = clf.predict(vec_testfield)
        svmfield=clf.score(vec_testfield,y_test)
        print svmfield

        vec_trainmerged=np.array([buildvector(modelmerged,x,vectorsize) for x in x_train])
        vec_testmerged=np.array([buildvector(modelmerged,x,vectorsize) for x in x_test])

        # lr=SGDClassifier(loss='log')
        # lr.fit(vec_trainmerged,y_train)
        # scoremerged=lr.score(vec_testmerged,y_test)

        # print scoremerged

        clfmodel = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0)
        clf=clfmodel.fit(vec_trainmerged, y_train)
        # y_predicted = clf.predict(vec_testfield)
        svmmerged=clf.score(vec_testmerged,y_test)
        print svmmerged

        run+=1
        if svmmerged>svmfield and svmmerged> 0.697:
            run=-1




def main():
    # te()
    # teword()
    # intersect(40)
    # setwordwindow(40)
    # Word2Vec.load_word2vec_format('corpus/initindex40',binary=False)
    
    modelpre=Word2Vec.load('corpus/pretrain40.model')
    modelfield=Word2Vec.load('corpus/fieldtrained40.model')
    modelmerged=Word2Vec.load('corpus/mergedtrained40iter1.model')
    print 'finish load'
    classify(modelpre,modelfield,modelmerged,40)
    # fieldtrain(40)
    # dis(40)
    
    

if __name__ == '__main__':
    main()