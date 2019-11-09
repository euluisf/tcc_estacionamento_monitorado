#!/usr/bin/python
# -*- coding: utf-8 -*
import cv2
import datetime
from class_Arquivo import Arquivo
from predicao import leituraPlaca
import warnings
import time

warnings.filterwarnings("ignore")
#TEMPO 00.50000

dici = {}
arq = Arquivo()
def reconhecePlacas_knn():
    pause = False

    cap = cv2.VideoCapture(0)
    carro_cascade = cv2.CascadeClassifier('cars.xml')
    placa_cascade = cv2.CascadeClassifier('plates_br.xml')
    while True: #Tentar voltar pra cá após detectar carro OK
        if pause == True:
            time.sleep(7) #Pausa 7 segundos pq achou um carro
            pause = False
        roda_video, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        carros = carro_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=1, minSize=(200, 200))


        for (x, y, w, h) in carros:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            placas = placa_cascade.detectMultiScale(roi_gray,scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))

            for (px, py, pw, ph) in placas:
                placaImg=roi_color.copy()
                cv2.rectangle(placaImg, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                placaImg = placaImg[py-5:py + ph+5, px-5:px + pw+5]
                inicio = datetime.datetime.now()
                placa=leituraPlaca(placaImg,"modelo.sav")#Chamando funcao de predicao da placa, 7 eh a certeza na precisao ( insira 1 ou 7)
                placaVazia=[]
                plate = ''.join(placa)
                if len(plate) == 7:
                    fim = datetime.datetime.now()
                    print('Tempo de Execução KNN: {}'.format(fim - inicio))
                #logPlacas(plate)
                print(plate[:3]+'-'+plate[3:])
                pause = True

        cv2.imshow('video', image)
        key = cv2.waitKey(1)
        if key == 27:
            break


def horaAtual():
    return datetime.now().strftime("%H:%M:%S")

def difHoras(time1, time2):
    time1 = time1.replace(':','')
    time2 = time2.replace(':','')
    return (int(time2) - int(time1)) # Resultado em Segundos

def logPlacas(texto):
    if texto in dici:
        log = dici[texto]
        atual = horaAtual()
        if (difHoras(log,atual) > 30):
            print('CARRO SAIU')
            dici[texto] = horaAtual()
            arq.updateLog(texto)
        else:
            pass
    else:
        #Criar entrada do carro no arquivo
        print('CARRO ENTROU')
        dici[texto] = horaAtual()
        arq.insertLog(texto)

if __name__ == '__main__':
    reconhecePlacas_knn()
