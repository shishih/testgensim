# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
import chardet
import sys     
reload(sys)  
sys.setdefaultencoding('utf-8') 

def wordclasscification():
    model=Word2Vec.load('corpus/mergedtrained40iter1.model')
    modelfield=Word2Vec.load('corpus/fieldtrained40.model')
    modelpre=Word2Vec.load('corpus/pretrain40.model')
    # wordlist=[u"喝酒",u"竞赛",u"原生",u"警察",u"离婚",u"单身"]
    with open('corpus/wordlabelcorpuslarge.txt') as fp:
        with open('corpus/wordneulabelsepe3','w') as file:
            for i in fp:
                # print i[:-1]
                try:
                    word=unicode(i[:-1])
                    upperline=0.016
                    floor=0.008 #0.01 0.013
                    upperlinefield=0.06
                    floorfield=0.02
                    upperlinepre=0.019
                    floorpre=0.018
                    try:
                        sub=(model.similarity(word,u"好")+model.similarity(word,u"快乐")+model.similarity(word,u"开心"))/3.0-(model.similarity(word,u"坏")+model.similarity(word,u"悲伤"))/2.0
                        if sub>upperline:
                            modellabel=1
                        elif sub<floor:
                            modellabel=-1
                        else:
                            modellabel=0
                        sub=(modelfield.similarity(word,u"好")+modelfield.similarity(word,u"快乐")+modelfield.similarity(word,u"开心"))/3.0-(modelfield.similarity(word,u"坏")+modelfield.similarity(word,u"悲伤"))/2.0
                        if sub>upperlinefield:
                            modelfieldlabel=1
                        elif sub<floorfield:
                            modelfieldlabel=-1
                        else:
                            modelfieldlabel=0
                        sub= (modelpre.similarity(word,u"好")+modelpre.similarity(word,u"快乐")+modelpre.similarity(word,u"开心"))/3.0-(modelpre.similarity(word,u"坏")+modelpre.similarity(word,u"悲伤"))/2.0
                        if sub>upperlinepre:
                            modelprelabel=1
                        elif sub<floorpre:
                            modelprelabel=-1
                        else:
                            modelprelabel=0
                        file.write(i[:-1]+' '+str(modellabel)+' '+str(modelfieldlabel)+' '+str(modelprelabel)+'\n') 
                    except KeyError:
                        print 'no key'
                        continue
                except UnicodeDecodeError:
                    print 'unicode error'
                    continue
            # for word in wordlist:
                # print word
                # word=i[:-1]
                # print word.encode('utf-8')
                
def codingdet():
    with open('corpus/wordtest.txt') as fp:
        for i in fp:
            print chardet.detect(i)

def main():
    wordclasscification()
    # codingdet()

if __name__ == '__main__':
    main()