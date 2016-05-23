

def main():
    with open('corpus/stopword.txt') as fp:
        data=fp.readlines()
        stopwordlist=[x[:-1] for x in data]
    with open('corpus/documentlabelstop.txt','a') as file:
        with open('corpus/documentlabel.txt','r') as fp:
            for row in fp:
                words=row.split(' ')
                file.write(words[0]+' ')
                for i in range(1,len(words)):
                    if words[i] in stopwordlist:
                        continue
                    else:
                        file.write(words[i]+' ')
                # file.write('\n')
if __name__ == '__main__':
    main()
