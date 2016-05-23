# -*-coding: utf-8 -*-

from collections import defaultdict
import pickle

def wordcount():
    dic=defaultdict(int)
    with open('corpus/answersinglenseg.txt') as fp:
        for row in fp:
            data=row[:-1].split()
            for word in data:
                dic[word]+=1
    sortlist=sorted(dic.items(),key=lambda x:x[1],reverse=True)
    # print sortlist[0][0],sortlist[0][1]
    with open('corpus/wordcount.dump','a') as fp:
        pickle.dump(sortlist,fp)
    with open('corpus/wordcount.txt','a') as fp:
        for word in sortlist:
            fp.write(word[0]+' '+str(word[1])+'\n')

def loadcount():
    with open('corpus/answersinglenseg.txt') as fp:
        data=pickle.load(fp)
        print data[0]
if __name__ == '__main__':
    wordcount()
    # loadcount()