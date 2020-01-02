print("RESOLUCIÓN MEDIANTE SIMPLEX DE PROBLEMA DE PROGRAMACION LINEAL")

import math

import numpy as np

funcionObjetivo = input("Introduce la funcion objetivo incluyendo TODAS las variables (ej: 1x1 + 2x2 +5x3 + 12x4): ")

max_min = input("¿Max/Min?: ").lower()

funcionObjetivo.strip()


#VARIABLES AUXILIARES


numeroVariablesOriginales=0

numeroVariables=0

contadorVariablesExtra=0

dosfases = False


for i in range(len(funcionObjetivo)):

	if funcionObjetivo[i] == "x":

		numeroVariables+=1

		numeroVariablesOriginales+=1



#INTRODUCCION DEL NUMERO DE RESTRICCIONES



numeroRestricciones="0"

while True:

	numeroRestricciones = input("Introduce el numero de restricciones: ")

	if numeroRestricciones.isdigit()==False:

		print("El valor introducido no es correcto. Inténtalo de nuevo")

	elif int(numeroRestricciones)==0:

		print("El numero de restricciones no puede ser 0. Inténtalo de nuevo")

	else:

		numeroRestricciones = int(numeroRestricciones)

		break




#CREACION DE LA MATRIZ DE RESTRICCIONES ORIGINAL



matrizRestricciones = []

for i in range(numeroRestricciones):

	restriccion = ""

	print("Introduce la restriccion numero ",i+1," : ")

	restriccion = input()

	matrizRestricciones.append(restriccion)




#CREACION DE LA MATRIZ DE RESTRICCIONES MODIFICADA CON VARIABLES DE HOLGURA Y AUXILIARES. CREACION DEL VECTOR B



vectorB = [0]*numeroRestricciones

for i in range(numeroRestricciones):

	for j in range(len(matrizRestricciones[i])):

		if matrizRestricciones[i][j] == "<":

			termino = matrizRestricciones[i].split("<")

			matrizRestricciones[i] = termino[0] + "+h" + str(i+1) + "<"  + termino[1]

			numeroVariables+=1

			if matrizRestricciones[i][j+1] == "-":

				vectorB[i] = int(termino[1].lstrip("-"))

			else:

				vectorB[i] = int(termino[1])

			break

		if matrizRestricciones[i][j]==">":

			dosfases = True

			termino = matrizRestricciones[i].split(">")

			matrizRestricciones[i] = termino[0] + "-h" + str(i+1) + "+a" + str(i+1) + ">" + termino[1]

			numeroVariables+=2

			if matrizRestricciones[i][j+1] == "-":

				vectorB[i] = int(termino[1].lstrip("-"))

			else:

				vectorB[i] = int(termino[1])

			break

		if matrizRestricciones[i][j]=="=":

			dosfases = True

			termino = matrizRestricciones[i].split("=")

			matrizRestricciones[i] = termino[0] + "+a" + str(i+1) + "=" + termino[1]

			numeroVariables+=1	

			if matrizRestricciones[i][j+1] == "-":

				vectorB[i] = int(termino[1].lstrip("-"))

			else:

				vectorB[i] = int(termino[1])

			break



#CREACION DEL VECTOR C



vectorC = [0]*numeroVariables

for i in range(len(funcionObjetivo)):

	if funcionObjetivo[i] == "x":

		valor_c = ""

		indice = ""

		k = i

		h = i

		while funcionObjetivo[h+1].isdigit():

			indice+=funcionObjetivo[h+1]

			if h+1==len(funcionObjetivo)-1:

				break

			else:

				h+=1

		indice_inver = indice[::-1]

		index = int(indice_inver)

		if k==0:

			vectorC[index-1] = 1

		else:

			while funcionObjetivo[k-1].isdigit():

				valor_c+=funcionObjetivo[k-1]

				k-=1

				if k==0:

					break

			valor_c_inver = valor_c[::-1]		

			if funcionObjetivo[k-1] == "-":

				if valor_c!="":

					vectorC[index-1] = -int(valor_c_inver)

				else:

					vectorC[index-1] = -1

			else:

				if valor_c!="":

					vectorC[index-1] = int(valor_c_inver)

				else:

					vectorC[index-1] = 1

fase2 = vectorC

fase1 = [0]*numeroVariables



#CREACION DE LA MATRIZ A



matrizA = []

for i in range(numeroRestricciones):

	matrizA.append([0]*numeroVariables)

for i in range(numeroRestricciones):

	for j in range(len(matrizRestricciones[i])):

		if matrizRestricciones[i][j] == "x":

			valor_a = ""

			indice = ""

			k = j

			h = j

			while matrizRestricciones[i][h+1].isdigit():

				indice+=matrizRestricciones[i][h+1]

				h+=1

				if h==len(matrizRestricciones[i]):

					break

			indice_inver = indice[::-1]

			index = int(indice_inver)

			if k==0:

				matrizA[i][index-1] = 1

			else:

				while matrizRestricciones[i][k-1].isdigit():

					valor_a+=matrizRestricciones[i][k-1]

					k-=1

					if k==0:

						break

				valor_a_inver = valor_a[::-1]		

				if matrizRestricciones[i][k-1] == "-":

					if valor_a!="":

						matrizA[i][index-1] = -int(valor_a_inver)

					else:

						matrizA[i][index-1] = -1

				else:

					if valor_a!="":

						matrizA[i][index-1] = int(valor_a_inver)

					else:

						matrizA[i][index-1] = 1


		if matrizRestricciones[i][j] == "a":

			contadorVariablesExtra+=1

			fase1[numeroVariablesOriginales + contadorVariablesExtra - 1] = -1

			matrizA[i][numeroVariablesOriginales + contadorVariablesExtra - 1] = 1

		if matrizRestricciones[i][j] == "h":

			k=j

			contadorVariablesExtra+=1

			fase1[numeroVariablesOriginales + contadorVariablesExtra - 1] = 0

			if matrizRestricciones[i][k-1] == "-":

				matrizA[i][numeroVariablesOriginales + contadorVariablesExtra - 1] = -1

			if matrizRestricciones[i][k-1] == "+":

				matrizA[i][numeroVariablesOriginales + contadorVariablesExtra - 1] = 1


print(matrizRestricciones)

print(matrizA)

print(vectorC)

print(fase1)

print(fase2)




#SALIDA POR PANTALLA DEL PROBLEMA REFORMULADO Y DE LOS VECTORES Y MATRICES ASOCIADOS


print("El problema reformulado quedaría así:")

print("Funcion Objetivo: ",funcionObjetivo)

print("Sujeto a:")

for i in range(numeroRestricciones):

	print(matrizRestricciones[i])

print("El vector de contribuciones unitarias al beneficio es: c = ",vectorC)

print("El vector de disponibilidad de recursos es: b = ",vectorB)

print("La matriz de coeficientes técnicos es:")

for i in range(numeroRestricciones):

	print(matrizA[i])





#MÉTODO DEL SIMPLEX



#CREO VECTORES Y MATRICES


#Creo Fase 1

vectorfase1 = np.zeros(numeroVariables)

for i in range(numeroVariables):

	vectorfase1[i] = fase1[i]

#Creo A


A = np.zeros([numeroRestricciones, numeroVariablesOriginales + contadorVariablesExtra])

for i in range(numeroRestricciones):

	for j in range(numeroVariablesOriginales + contadorVariablesExtra):

		A[i][j]=matrizA[i][j]


#Creo Cb


Cb = np.zeros(numeroRestricciones)

for i in range(numeroRestricciones):

	Cb[i] = vectorC[numeroVariablesOriginales + i]


#Creo C


C = np.zeros(numeroVariablesOriginales + contadorVariablesExtra)

for i in range(numeroVariablesOriginales + contadorVariablesExtra):

	C[i] = vectorC[i]


#Creo b


b = np.zeros(numeroRestricciones)

for i in range(numeroRestricciones):

	b[i] = vectorB[i]


#Creo Ub (vector de nivel de realizacion de las variables básicas)


Ub = np.zeros(numeroRestricciones)


#Creo Vb (vector de criterios del simplex de las variables)


Vb = np.zeros(numeroVariablesOriginales + contadorVariablesExtra)


#Creo Pb (matriz de tasas de sustitucion)


Pb = np.zeros([numeroRestricciones, numeroVariablesOriginales + contadorVariablesExtra])


#Creo B (Base)


B = np.identity(numeroRestricciones)

#for i in range(numeroRestricciones):

	#for j in range(numeroRestricciones):

		#B[i][j] = A[i][numeroVariablesOriginales + j]


#Creo Z (Valor funcion objetivo)


Z = np.zeros(1)


#Creo precios sombra de restricciones


ps = np.zeros(numeroRestricciones)




#RESOLUCION FASE 1

if dosfases == True:

	indices_sumar, = np.where(vectorfase1 == -1)

	for i in indices_sumar:

		for j in range(numeroRestricciones):

			if A[j][i] == 1:

				vectorfase1 = vectorfase1 + A[j]


	Vb = vectorfase1

	Ub = b

	Pb = A


	optimo = False

	while optimo == False:

		optimo = True


		#ELECCIÓN DE VARIABLES QUE ENTRAN Y SALEN


		indiceVariableEntra = 0

		for i in range(len(Vb)):

			if max_min == "max":

				if Vb[i]>Vb[indiceVariableEntra]:

					indiceVariableEntra = i

			else:

				if Vb[i]<Vb[indiceVariableEntra]:

					indiceVariableEntra = i


		resultadoDivisionmax = math.inf

		resultadoDivisionmin = 0

		indice_Ub_VariableSale = 0

		for i in range(numeroRestricciones):

			if Pb[i][indiceVariableEntra]>0:

				if max_min == "max":

 					if Ub[i]/Pb[i][indiceVariableEntra]<resultadoDivisionmax and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmax = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i

				
				else:

 					if Ub[i]/Pb[i][indiceVariableEntra]>resultadoDivisionmin and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmin = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i




		print(Ub)
 		
		print(Vb)

		print(Pb)			


		#CREO NUEVA BASE

 	
		for i in range(numeroRestricciones):

			B[i][indice_Ub_VariableSale] = A[i][indiceVariableEntra]


		#CREO Cb

		for i in range(numeroRestricciones):

			Cb[indice_Ub_VariableSale] = vectorC[indiceVariableEntra]


		#CREO B INVERSA

		print(B)

		inv_B = np.linalg.inv(B)


		#CREO NUEVAS Ub, Vb, Pb, Z y ps


		Vb = C - np.dot(np.dot(Cb,inv_B),A)

		Ub = np.dot(inv_B,b)

		Pb = np.dot(inv_B,A)

		Z = np.dot(np.dot(Cb,inv_B),b)

		ps = np.dot(Cb,inv_B)
	

		#COMPRUEBO OPTIMALIDAD


		for i in range(numeroVariablesOriginales + contadorVariablesExtra):

			if Vb[i]>0:

				optimo = False


	#RESOLUCION FASE 2



	#SOLUCION FACTIBLE INICIAL

	vectorfase2 = np.zeros(numeroRestricciones)

	vectorfase2 = C - np.dot(np.dot(Cb,inv_B),A)


	Vb = vectorfase2


	optimo = False

	while optimo == False:

		optimo = True


		#ELECCIÓN DE VARIABLES QUE ENTRAN Y SALEN


		indiceVariableEntra = 0

		for i in range(len(Vb)):

			if max_min == "max":

				if Vb[i]>Vb[indiceVariableEntra]:

					indiceVariableEntra = i

			else:

				if Vb[i]<Vb[indiceVariableEntra]:

					indiceVariableEntra = i


		resultadoDivisionmax = math.inf

		resultadoDivisionmin = 0

		for i in range(numeroRestricciones):

			if Pb[i][indiceVariableEntra]>0:

				if max_min == "max":

 					if Ub[i]/Pb[i][indiceVariableEntra]<resultadoDivisionmax and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmax = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i

				else:

 					if Ub[i]/Pb[i][indiceVariableEntra]>resultadoDivisionmin and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmin = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i

		print(Ub)
 		
		print(Vb)

		print(Pb)			

		#CREO NUEVA BASE

 	
		for i in range(numeroRestricciones):

			B[i][indice_Ub_VariableSale] = A[i][indiceVariableEntra]


		#CREO Cb

		for i in range(numeroRestricciones):

			Cb[indice_Ub_VariableSale] = vectorC[indiceVariableEntra]


		#CREO B INVERSA


		inv_B = np.linalg.inv(B)


		#CREO NUEVAS Ub, Vb, Pb, Z y ps


		Vb = C - np.dot(np.dot(Cb,inv_B),A)

		Ub = np.dot(inv_B,b)

		Pb = np.dot(inv_B,A)

		Z = np.dot(np.dot(Cb,inv_B),b)

		ps = np.dot(Cb,inv_B)
	

		#COMPRUEBO OPTIMALIDAD


		for i in range(numeroVariablesOriginales + contadorVariablesExtra):

			if Vb[i]>0:

				optimo = False


	#SALIDA POR PANTALLA DE LA SOLUCIÓN ÓPTIMA


	print("El valor máximo que puede alcanzar la función objetivo es: ",Z)

	print("Y se alcanza con el siguiente mix de producción: ", Ub)

	print("El precio sombra de los recursos es: ",ps)

else:

	Vb = C

	Ub = b

	Pb = A

	factible = False

	optimo = False

	while optimo == False and factible ==False:

		optimo = True

		factible = True

		#ELECCIÓN DE VARIABLES QUE ENTRAN Y SALEN


		indiceVariableEntra = 0

		for i in range(len(Vb)):

			if max_min == "max":

				if Vb[i]>Vb[indiceVariableEntra]:

					indiceVariableEntra = i

			else:

				if Vb[i]<Vb[indiceVariableEntra]:

					indiceVariableEntra = i


		resultadoDivisionmax = math.inf

		resultadoDivisionmin = 0

		for i in range(numeroRestricciones):

			if Pb[i][indiceVariableEntra]>0:

				if max_min == "max":

 					if Ub[i]/Pb[i][indiceVariableEntra]<resultadoDivisionmax and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmax = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i

				else:

 					if Ub[i]/Pb[i][indiceVariableEntra]>resultadoDivisionmin and Ub[i]/Pb[i][indiceVariableEntra]>0:

 						resultadoDivisionmin = Ub[i]/Pb[i][indiceVariableEntra]

 						indice_Ub_VariableSale = i

		print(Ub)
 		
		print(Vb)

		print(Pb)			

		#CREO NUEVA BASE

 	
		for i in range(numeroRestricciones):

			B[i][indice_Ub_VariableSale] = A[i][indiceVariableEntra]


		#CREO Cb

		for i in range(numeroRestricciones):

			Cb[indice_Ub_VariableSale] = vectorC[indiceVariableEntra]


		#CREO B INVERSA


		inv_B = np.linalg.inv(B)


		#CREO NUEVAS Ub, Vb, Pb, Z y ps


		Vb = C - np.dot(np.dot(Cb,inv_B),A)

		Ub = np.dot(inv_B,b)

		Pb = np.dot(inv_B,A)

		Z = np.dot(np.dot(Cb,inv_B),b)

		ps = np.dot(Cb,inv_B)
	

		#COMPRUEBO OPTIMALIDAD


		for i in range(numeroVariablesOriginales + contadorVariablesExtra):

			if Vb[i]>0:

				optimo = False

		for i in range(numeroRestricciones):

			if Ub[i]<0:

				factible = False


	#SALIDA POR PANTALLA DE LA SOLUCIÓN ÓPTIMA


	print("El valor máximo que puede alcanzar la función objetivo es: ",Z)

	print("Y se alcanza con el siguiente mix de producción: ", Ub)

	print("El precio sombra de los recursos es: ",ps)