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
import chardet

import sys     
reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入     
sys.setdefaultencoding('utf-8')  

def buildvector(model,x,vectorsize):
    # size=x.length
    vec=np.zeros(vectorsize)
    count=0.0
    for i in x:
        try:
            vec+=model[unicode(i)]
            count+=1.0
        except KeyError:
            # print 'keyerror'
            continue
    if count!=0.0:
        vec/=count
        # print vec
    return vec
def main():
    modelpre=Word2Vec.load('corpus/pretrain40.model')
    # modelpre=Word2Vec.load_word2vec_format('corpus/pretrain40',binary=False)
    with open('corpus/documentlabel.txt') as fp:
        # print fp.readlines()[0]
        xl=[]
        yl=[]
        for i in fp:
            yl.append(int(i[0]))
            xl.append(i[2:-1].split(' ')) 
    y=np.array(yl)
    x=np.array(xl)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    # print x
    # print x_train[0][0]

    # print modelpre[unicode(x_train[0][0])]
    vectorsize=40
    vec_trainpre=np.array([buildvector(modelpre,x,vectorsize) for x in x_train])



if __name__ == '__main__':
    main()
    # x='你好'
    # print type(x)
    # ux=unicode(x)
    # print type(ux)