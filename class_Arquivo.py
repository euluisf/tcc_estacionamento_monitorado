import datetime

class Arquivo(object):

    def __init__(self):
        self.dicionario = {}
        self.get_log()

    def get_log(self):
        arq = open('Registro.txt','w+')
        arq.close()
        arq =open('Registro.txt','r')
        for line in arq:
            for i in range(len(line)):
                if line[i] == ';':
                    self.dicionario[line[0:i]] = line[i+1:-1]
        return self.dicionario

    def updateLog(self, string): #carro saiu
        now = datetime.datetime.now()
        temp = self.dicionario[string]
        del self.dicionario[string]
        time = "%s:%s:%s" %(now.hour, now.minute, now.second)
        new_log = (temp + " - " + time)
        self.dicionario[string] = new_log
        self.saveLog()

    def insertLog(self, string): #carro entrou
        now = datetime.datetime.now()
        time = "%s:%s:%s" %(now.hour, now.minute, now.second)
        self.dicionario[string] = time
        self.saveLog()

    def saveLog(self):
        arq = open('Registro.txt','w+')
        for i in self.dicionario:
            arq.write(i + ';' + self.dicionario[i])
            arq.write('\n')
        arq.close()
