import numpy as np
import cv2
import pickle

#==================== EDICOES BASICAS PARA A IMAGEM ====================
#PARAMETROS DE ENTRADA: imagem(imagem)
#RETORNO: imagem em tons de cinza (imagemCinza)
def editaPlaca(imagem):
	imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
	imagemCinza=cv2.equalizeHist(imagemCinza)
	return imagemCinza

#==================== FUNCAO DE ORDENACAO DOS CONTORNOS ====================
#PARAMETROS DE ENTRADA: contornos localizados (cnts)
#RETORNO: contornos porem agora ordenados em relacao ao eixo X (esquerda para direita)
'''
A função de ordenação de contornos foi baseada na disponibilizada no site: 
https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
'''
def ordenaContornos(cnts):
	reverse = False
	i = 0
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
	return (cnts)# retorna a lista de contornos ordenada

#==================== FUNCAO DE BUSCA DOS CARACTERES DA PLACA ====================
#PARAMETROS DE ENTRADA: imagem (imagem), valor divisorio da funcao thresh (threshN), numero de caracteres que se deseja encontrar (nCaracteresDesejado)
#RETORNO: a funcao retorna o numero de caracteres encontrados (numeroCaracteres) e um vetor contendo as coordenadas destes caracteres (posCaracteres)
def procuraCaracteres(imagem,threshN,nCaracteresDesejado):
	#a funcao de threshold binariza a imagem somente deixando os pixels com 255 ou 0, o parametro threshN eh o valor de divisao para o pixel virar 0 ou 255
	_,threshold = cv2.threshold(imagem,threshN,255,cv2.THRESH_BINARY)
	_,contours,hierarchy= cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	posCaracteres=np.zeros((nCaracteresDesejado,4))
	posCaracteresTemp=[]#Variavel que ira conter os contornos localizados na triagem inicial
	#========= buscando contorno correto da placa(sera possivelmente o maior contorno dentro da imagem)
	alturaPlaca=0
	larguraPlaca=0
	for cnt in contours:
		[x, y, w, h] = cv2.boundingRect(cnt)
		if h>alturaPlaca:
			alturaPlaca=h
			larguraPlaca=w
	multiplicadorAltura=0.35
	multiplicadorLargura=0.25
	#==== Ordenando os contornos localizados
	try:
		contours=ordenaContornos(contours) #ordenando os contornos localizados
	except Exception as e:
		print(e)

	#========= Buscando os caracteres da placa
	#Esta eh somente uma triagem inicial dos contornos localizados na placa
	numeroCaracteresTemp=0
	for cnt in contours:
		[x,y,w,h] = cv2.boundingRect(cnt)
		#=== definindo o tamanho dos contornos que podem ser possiveis caracteres
		if h>=(alturaPlaca*multiplicadorAltura) and w<larguraPlaca*multiplicadorLargura:
			try:
				posCaracteresTemp.append([x,y,w,h])
			except:
				numeroCaracteresTemp=0
				break
			numeroCaracteresTemp+=1
	caracterAtual=0 #e tambem o numero de caracteres localizado
	#=== Agora eh verificado qual dos contornos realmente sao caracteres(isso eh feito com base na distancia entre um contorno e outro e seus tamanhos)
	if numeroCaracteresTemp>=7:
		for i in range(0,numeroCaracteresTemp):
			#== Verificando se esta na primeira iteracao para comparar o contorno atual com o proximo, ambos possiveis caracteres
			if caracterAtual==0 :
				x=(int)(posCaracteresTemp[i][0])
				w=(int)(posCaracteresTemp[i][2])
				y=(int)(posCaracteresTemp[i][1])
				h=(int)(posCaracteresTemp[i][3])
				somador=1
			#== Nas demais iteraçoes ira comparar o ultimo caracter localizado com o proximo contorno(possivel caracter)
			else:
				x=(int)(posCaracteres[caracterAtual-1][0])
				w=(int)(posCaracteres[caracterAtual-1][2])
				y=(int)(posCaracteres[caracterAtual-1][1])
				h=(int)(posCaracteres[caracterAtual-1][3])
				somador=0
			#= estes multiplicador eh para dar a distancia entre o ponto inicial de um caracter ate o proximo
			multiplicadorDistancia=1.5
			#= caso seja o terceiro caracter ou o caracter seja "1" ou "I"
			if caracterAtual == 3 or w*2<=posCaracteresTemp[i+somador][2]or w*1.6<posCaracteres[caracterAtual+somador-2][2]:
				multiplicadorDistancia=5
			try:
			#== O if compara o caracter atual com o proximo possivel caracter para ver se eh realmente um caracter ou algum contorno enganoso
			#as comparacoes que ele faz sao com base nas relacoes de posicionamento entre o caracter anterior e o proximo
			#DESENHAR ESTE IF EXPLICANDO PARA MONO
				if x+multiplicadorDistancia*w>=posCaracteresTemp[i+somador][0] and  (y-0.2*h)<posCaracteresTemp[i+somador][1] and (y+h*0.2)>posCaracteresTemp[i+somador][1] and ((y+h)*0.85)<(posCaracteresTemp[i+somador][1]+posCaracteresTemp[i+somador][3]) and (1.15*(y+h))>(posCaracteresTemp[i+somador][1]+posCaracteresTemp[i+somador][3]) :

					try:
						posCaracteres[caracterAtual]=posCaracteresTemp[i]
						caracterAtual+=1
					except: #excedeu de 7 caracteres localizados
						break
			except:
				break
	numeroCaracteres=caracterAtual
	return numeroCaracteres,posCaracteres

#==================== FUNCAO DE PREDICAO ====================
#Função que ira predizer o caracter encontrado por meio do modelo KNN ja treinado
#PARAMETROS DE ENTRADA: imagem(imagem), posicao do caracter em questao(posCaracter), modelo treinado(modelo)
#RETORNO: Caracter predito (chr(int(predict))), score do caracter (np.amax(y_scores) )
def predicao(imagem,posCaracter,modelo):
	#==== obtendo a posicao da imagem
	x=(int)(posCaracter[0])
	y=(int)(posCaracter[1])
	w=(int)(posCaracter[2])
	h=(int)(posCaracter[3])
	#==== cortando a imagem
	caracter = imagem[y:y+h, x:x+w]
	caracterRes = cv2.resize(caracter,(7,12))
	caracterRes = caracterRes.reshape((1,84))
	caracterRes = np.float32(caracterRes)
	#==== predicao
	predict=modelo.predict(caracterRes)# 3 vizinho, ou seja os 3 mais proximos caracteres
	y_scores = modelo.predict_proba(caracterRes) # 3 caracteres T -> I , T , T => 66% scores
	# Se houver 20% de chances de ser o Q e menos de 50% de chances de serem o 0 ou O retorna Q
	for i in range (len(y_scores)):
		if ((y_scores[i][26] >= 0.2) and (y_scores[i][0] < 0.5) and y_scores[i][24] < 0.5):
			return 'Q',np.amax(y_scores)
	return chr(int(predict)),np.amax(y_scores)

#==================== FUNCAO DE LEITURA DA PLACA ====================
#PARAMETROS DE ENTRADA: imagem(imagem) e o local onde se encontra  modelo treinado(modelo)
#RETORNO: lista de caracteres(placa)

def leituraPlaca(imagem,modelo):
	#==== carregando o modelo que ira realizar a predição
	loaded_model = pickle.load(open(modelo, 'rb'))
	imagem=editaPlaca(imagem)#recebe a imagem ja editada

	#================ Buscando os caracteres da placa
	# parametros utilizados na busca dos caracteres
	nDesejados=7								#numero de caracteres que se deseja localizar
	passo=2										#tamanho do passo do threshold para cada iteração (quanto maior mais rapido o algoritmo e menos preciso)
	tentativas=0								#tentativas da placa atual
	tantativasTotais=190/passo					#tentativas totais para a placa atual
	parametroThresh=25							#Valor inicial para o ThreshHold
	maxScore=0									#Valor maximo de score para a placa
	placaM=[]
	scoreLetras=np.zeros((7,36))
	#Neste momento eh feito a busca dos caracteres na tela por meio dos contornos para varios tons diferentes ate que se localize os 7 caracteres
	scorePlaca=0
	while(tentativas < tantativasTotais and scorePlaca <= 7):
		scorePlaca=0
		nEncontrados,posCaracteres=procuraCaracteres(imagem,parametroThresh,nDesejados)
		if (nEncontrados==nDesejados ):
			placa=[]
			for i in range(nEncontrados):
				caracter,score=predicao(imagem,posCaracteres[i],loaded_model) #aqui eh o principal, onde esta predizendo cada caracter pelo passo inferior
				valorEmAsc=ord(caracter)
				if valorEmAsc<65: # numeros
					scoreLetras[i][valorEmAsc-48]+=score#somando o score para este caracter nesta posicao
				else:
					scoreLetras[i][valorEmAsc-55]+=score
				scorePlaca=scorePlaca+score
				placa.append(caracter)
			#para o caso de a certezaPredicao for menor que 7 e o algoritmo encontrar mais de uma placa com 7 caracteres ele ira retornar a que tiver obtido maior pontuacao
			if scorePlaca> maxScore:
				maxScore=scorePlaca
				placaM=placa
			#Sempre que forem achados 7 caracteres o valor de tentativas eh aumentado pois foi uma tentativa bem sucedida e portanto nao precisamos repetir muitas vezes este processo de localizacao de caracteres
			tentativas+=10
		parametroThresh=parametroThresh+passo # 25 + 3 = 28
		tentativas+=1
	scorePos=[0]*7
	letra=[0]*7

	#Verifica para cada posicao da placa o caracter que obteve maior pontuacao nela
	for i in range(0,36):#Letras do alfabeto + 0 a 9
		for k in range(0,7):
			if scoreLetras[k][i]>=scorePos[k]:
				scorePos[k]=scoreLetras[k][i]
				letra[k]=i #caractere
	if placaM!=[]:
		for i in range(0,7):
			if letra[i]<10:
				placaM[i]=str(chr(letra[i]+48))
			else:
				placaM[i]=str(chr(letra[i]+55))

	placaM=limparPlaca(placaM)

	return placaM

#==================== Função de correção da placa ====================
#PARAMETROS DE ENTRADA: lista de caracteres localizados(placa)
#RETORNO: lista de caracteres ja transformados(letras + numeros)

'''
Função retirada e adaptada do trabalho
ANTONELLO, L. L. R. Identificação automática de placa de veículos através de
processamento de imagem e visão computacional. 2017.
'''
def limparPlaca(placa):
	letras = placa[:3]
	numeros = placa[3:]
	letras = [w.replace('0', 'O') for w in letras]
	letras = [w.replace('1', 'I') for w in letras]
	letras = [w.replace('4', 'A') for w in letras]
	letras = [w.replace('6', 'G') for w in letras]
	letras = [w.replace('7', 'I') for w in letras]
	letras = [w.replace('8', 'B') for w in letras]
	letras = [w.replace('2', 'Z') for w in letras]
	letras = [w.replace('5', 'S') for w in letras]

	numeros = [w.replace('B', '8') for w in numeros]
	numeros = [w.replace('T', '1') for w in numeros]
	numeros = [w.replace('Z', '2') for w in numeros]
	numeros = [w.replace('S', '5') for w in numeros]
	numeros = [w.replace('A', '4') for w in numeros]
	numeros = [w.replace('O', '0') for w in numeros]
	numeros = [w.replace('I', '1') for w in numeros]
	numeros = [w.replace('G', '6') for w in numeros]
	numeros = [w.replace('Q', '0') for w in numeros]
	numeros = [w.replace('D', '0') for w in numeros]
	return letras + numeros