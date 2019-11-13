# -*- coding: utf-8 -*
import cv2
import datetime
import pytesseract as ocr
import re
from predicao import leituraPlaca
import time

current_milli_time = lambda: int(round(time.time() * 1000))

'''
 Modelo XML(plates_br.xml) disponibilizado pelo autor Leonardo Leite (https://docplayer.com.br/73452367-Identificacao-automatica-de-placa-de-veiculos-atraves-de-processamento-de-imagem-e-visao-computacional.html)
 Modelo XML(cars.xml) retirado de: https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml
'''
carro_cascade = cv2.CascadeClassifier('cars.xml')
placa_cascade = cv2.CascadeClassifier('plates_br.xml')
log = ''
media, cont, det_certa, det_errada, det_branco = 0,0,0,0,0
sf = 1.04
sfp = 1.01
arq = open('Testes_Haar_Placa.txt','a')
arq.write('Teste com Scale Factor : '+str(sfp))
arq.write('\n')
arq.write('\n')
arq.close()


for i in range(1,31):
    if i < 10:
        nome = 'track0093[0'+str(i)+'].png'
    else:
        nome = 'track0093['+str(i) + '].png'
    image = cv2.imread('testes/' + nome)
    #placa_cascade = cv2.CascadeClassifier('plates_br.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    carros = carro_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=(1), minSize=(300, 300))



    for (x, y, w, h) in carros:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        inicio = current_milli_time()
        placas = placa_cascade.detectMultiScale(roi_gray, scaleFactor=sfp, minNeighbors=7,  maxSize=(100, 100))
        fim = current_milli_time()
        for (px, py, pw, ph) in placas:
            placaImg = roi_color.copy()
            cv2.rectangle(roi_color, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
            cont += 1

    W = 700.
    oriimg = image
    height, width, depth = oriimg.shape
    imgScale = W / width
    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    #cv2.imshow("Show by CV2", newimg)
    key = cv2.waitKey(1)

    media += (fim - inicio)

    if cont == 1:
        det_certa += 1
    elif cont == 0:
        det_branco += 1
    else:
        det_errada += 1
    log = (nome + ' --- ' + str(fim - inicio) + '(ms)' + ' --- ' + 'Numero Detecções: ' + str(cont))
    arq = open('Testes_Haar_Placa.txt','a')
    arq.write(log)
    arq.write('\n')
    if i == 30:
        arq.write('\n')
        arq.write('Media(ms) : '+ str(media / 30))
        arq.write('\n')
        arq.write('Detecções Corretas : ' + str(det_certa))
        arq.write('\n')
        arq.write('Detecções Incorretas : ' + str(det_errada))
        arq.write('\n')
        arq.write('Imagens sem Detecções : ' + str(det_branco))
        arq.write('\n')
        arq.write('\n')
        arq.write('--------------------------------------------------------')
        arq.write('\n')
        arq.write('\n')
    arq.close()
    cont = 0
