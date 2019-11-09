#!/usr/bin/python
# -*- coding: utf-8 -*

import cv2
import pytesseract as ocr
import datetime
import re
from class_Arquivo import Arquivo
from predicao import leituraPlaca, limparPlaca

import time

current_milli_time = lambda: int(round(time.time() * 1000))

dici = {}
arq = Arquivo()
ocr.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def reconhecePlacas(video, algoritmo):
    pausaDeteccao, contaErro, contaAcerto, somador, wait = 0, 0, 0, 0, 0

    inicio = datetime.datetime.now()


    placa_cascade = cv2.CascadeClassifier('plates_br.xml')
    carro_cascade = cv2.CascadeClassifier('cars.xml')

    cap = cv2.VideoCapture(video)
    roda_video, image = cap.read()

    while roda_video:
        frame = int(round(cap.get(1)))
        roda_video, image = cap.read()
        pausaDeteccao = contaAcerto

        if not roda_video:
            break

        if wait > 0:  # Condição para pausar
            wait -= 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        inicio_haar = current_milli_time()
        carros = carro_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=2, minSize=(500, 500))

        for (x, y, w, h) in carros:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]



            placas = placa_cascade.detectMultiScale(roi_gray, scaleFactor=1.04, minNeighbors=7, minSize=(30, 30),maxSize=(100, 100))

            for (px, py, pw, ph) in placas:
                placaImg = roi_color.copy()
                cv2.rectangle(roi_color, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                img = roi_color[py:py + ph, px:px + pw]
                placaImg = placaImg[py - 5:py + ph + 5, px - 5:px + pw + 5]

                if algoritmo == 'knn':
                    placa = leituraPlaca(placaImg, "modelo.sav")  # Chamando funcao de predicao da placa
                    placaInvalida = []
                    somador += 1

                    if placa == placaInvalida:
                        contaErro += 1
                    else:
                        contaAcerto += 1
                        plate = ''.join(placa)
                        plate = (plate[:3] + '-' + plate[3:])
                        print(plate)

                elif algoritmo == 'ocr':
                    somador += 1
                    ocr_ini = datetime.datetime.now()
                    texto = ocr.image_to_string(img, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789')
                    texto = re.sub('[^a-zA-Z0-9 \\\]', '', texto)
                    if texto.isalnum():
                        if len(texto) == 7:
                            contaAcerto += 1
                            texto = limparPlaca(texto)
                            plate = ''.join(texto)
                            plate = (plate[:3] + '-' + plate[3:])
                            print(plate)
                            ocr_fim = datetime.datetime.now()
                            print("Tempo Digitalização com ocr: {} ".format(ocr_fim-ocr_ini))

                        else:
                            contaErro += 1
                    else:
                        contaErro += 1

                else:
                    print('Parâmetros Inválidos')

        W = 1200.
        oriimg = image
        height, width, depth = oriimg.shape
        imgScale = W / width
        newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
        newimg = cv2.resize(oriimg, (int(newX), int(newY)))

        cv2.imshow("Deteccao de Placas", newimg)
        key = cv2.waitKey(1)
        if key == 27:
            break

        if pausaDeteccao < contaAcerto:  #Pausa o algoritmo e Salva LOG
            logPlacas(plate, frame) #salva os registros num arquivo a parte log
            wait = 120  # 4 segundos

    cap.release()
    cv2.destroyAllWindows()
    fim = datetime.datetime.now()

    print('Tempo de Execução Algoritmo {}: {}'.format(algoritmo, fim - inicio))
    print('Total de Detecções Corretas: ' + str(contaAcerto))
    print("Total de Detecções Inválidas: " + str(contaErro))
    print("Total de Imagens Testadas: " + str(somador))

def logPlacas(texto, frame_atual):
    qntd = 450  # 15 segundos, tempo para o carro passar pela portaria e decidir se volta ou nao
    if texto in dici:
        if (frame_atual - dici[texto] > qntd):
            # Carro Saindo
            arq.updateLog(texto)
            dici[texto] = frame_atual
        else:
            # Sobreescreve o dicionario com o novo frame
            dici[texto] = frame_atual
    else:
        # Criar entrada do carro no arquivo
        dici[texto] = frame_atual
        arq.insertLog(texto)


if __name__ == '__main__':
    reconhecePlacas('..\Luis.mp4','knn')