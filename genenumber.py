
def main():
    with open('corpus/answerjiebaseg.txt') as fp:
        data=fp.readlines()
    
    num=1
    for i in data:
        no=num/10000
        with open('corpus/tobetag'+str(no)+'.txt','a') as fp:
            fp.write(str(num)+' '+i.strip('\r'))
            num+=1

if __name__ == '__main__':
    main()