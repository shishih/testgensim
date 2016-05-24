# -*-coding: utf-8 -*-

from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
import numpy as np
from palettable.colorbrewer.qualitative import Dark2_8
from palettable.colorbrewer.qualitative import Pastel2_5
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from pylab import *
import seaborn as sns
import sys     
reload(sys)  
sys.setdefaultencoding('utf-8') 

def gene():
    modelpre=Word2Vec.load('corpus/pretrain40.model')
    modelfield=Word2Vec.load('corpus/fieldtrained40.model')
    modelmerged=Word2Vec.load('corpus/mergedtrained40iter1.model')
    xlist=[]
    ylist=[]
    zlist=[]
    labellist=[]
    upperline=0.016
    floor=0.008 #0.01 0.013
    upperlinefield=0.06
    floorfield=0.02
    upperlinepre=0.019
    floorpre=0.018
    with open('corpus/word2pic2.txt') as fp:
        for row in fp:
            word=unicode(row[:-1])
            x=(modelmerged.similarity(word,u"好")+modelmerged.similarity(word,u"快乐")+modelmerged.similarity(word,u"开心"))/3.0-(modelmerged.similarity(word,u"坏")+modelmerged.similarity(word,u"悲伤"))/2.0
            y=(modelfield.similarity(word,u"好")+modelfield.similarity(word,u"快乐")+modelfield.similarity(word,u"开心"))/3.0-(modelfield.similarity(word,u"坏")+modelfield.similarity(word,u"悲伤"))/2.0
            z=(modelpre.similarity(word,u"好")+modelpre.similarity(word,u"快乐")+modelpre.similarity(word,u"开心"))/3.0-(modelpre.similarity(word,u"坏")+modelpre.similarity(word,u"悲伤"))/2.0
            labellist.append(word)
            # xlist.append(x-(upperline+floor)/2.0)
            xlist.append(x-0.016)
            ylist.append(y-(upperlinefield+floorfield)/2.0)
            zlist.append(z-(upperlinepre+floorpre)/2.0)
    # with open('corpus/word2picxyz.txt','w') as fp:
    #     pickle.dump(labellist,xlist,ylist,zlist,fp)
    return labellist,xlist,ylist,zlist

def draw(labellist,xlist,ylist,zlist):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # myfont = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/msyi.ttf')  
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False  

    for label, x, y, z in zip(labellist,xlist,ylist,zlist):
        # label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir) + "haha"
        # if (x<0 and y>0 and z>0):
        ax.text(x*10, y*10, z*10, str(label), None) # zdir means the direction of label
        # ax.text(x*10, y*10, z*10, str(label)+str(x)[:1]+str(y)[:1]+str(z)[:1], None) # zdir means the direction of label

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.set_xlabel('model 3')
    ax.set_ylabel('model 2')
    ax.set_zlabel('model 1')
    
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

    # ax.title("代表词汇在三个模型中的分布")

    plt.show()

if __name__ == '__main__':
    labellist,xlist,ylist,zlist=gene()
    draw(labellist,xlist,ylist,zlist)