# -*- coding:utf8 -*-

def main():
    with open('corpus/newpositive') as fp:
        data=fp.readlines()
        print len(data)
        
    with open('corpus/newnegative') as fp:
        data2=fp.readlines()
        print len(data2)
        print len(data)+len(data2)
    data.extend(data2)
    datasum=set(data)
    print len(datasum)
    # with open('corpus/newnegative','a') as fp:
    #     for word in data:
    #         fp.write(word)
if __name__ == '__main__':
    main()