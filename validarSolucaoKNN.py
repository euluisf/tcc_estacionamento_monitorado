from predicao import leituraPlaca
import cv2

somador, contaErro, contaAcerto = 0,0,0

for i in range(1,21):
    if i < 10:
        nome = 'placa0'+str(i)+'.jpg'
    else:
        nome = 'placa'+str(i) + '.jpg'

    placaImg = cv2.imread('validacaoKNN/' + nome)

    cv2.imshow("Placa", placaImg)
    key = cv2.waitKey(1)
    if key == 27:
        break
    placa = leituraPlaca(placaImg, "modelo.sav")  # Chamando funcao de predicao da placa
    placaInvalida = []
    somador += 1

    if placa == placaInvalida:
        contaErro += 1
        plate = 'NAO DETECTADA'
    else:
        contaAcerto += 1
        plate = ''.join(placa)
        plate = (plate[:3] + '-' + plate[3:])
        print(plate)

    arq = open('validacaoKNN.txt', 'a')
    arq.write(nome + '----' + plate)
    arq.write('\n')
    arq.close()
