import tkinter as tk

from tkcalendar import DateEntry

import mysql.connector

mydb = mysql.connector.connect(host = "localhost", user = "root", password = "x-mania1998", database = "gym")

mycursor = mydb.cursor()

class root(tk.Tk):

	def __init__(self):

		tk.Tk.__init__(self)

		self.title("Gym")
		self.geometry("300x150")
		self.resizable(0,0)

def open_window(name, lista, color1, color2, color3):

	if name == "push":

		ventana_push = ventana_ejercicios(name)

		container_push = menu_ejercicios(ventana_push, lista, name, color1, color2, color3)
		container_push.pack(expand = True, fill = "both")

		ventana_push.mainloop()

	if name == "pull":

		ventana_pull = ventana_ejercicios(name)

		container_pull = menu_ejercicios(ventana_pull, lista, name, color1, color2, color3)
		container_pull.pack(expand = True, fill = "both")

		ventana_pull.mainloop()

	if name == "legs":

		ventana_legs = ventana_ejercicios(name)

		container_legs = menu_ejercicios(ventana_legs, lista, name, color1, color2, color3)
		container_legs.pack(expand = True, fill = "both")

		ventana_legs.mainloop()


class menu_principal(tk.Frame):

	def __init__(self, parent):

		tk.Frame.__init__(self, parent, bg = "red")

		lista_push = ["Bench Press", "Incline Dumbbell Press", "Cable Fly", 
		"Lateral Raises", "Skull Crusher", "Cable Pushdown"]

		lista_pull = ["Bent Over Barbell Row", "Pull Ups", "T-Bar Row", "Rear Delt Isolation", 
		"Standing Barbell Curl", "Dumbbell Curl", "Shrugs"]

		lista_legs = ["Squats", "Romanian Deadlifts", "Leg Press", "Leg Curls", "Standing Calves", "Seated Calves"]

		color_push = ["#90f0ed", "#13d4ce", "#1397d4"]

		color_pull = ["#f5ae78", "#f28a3a", "#de5a21"]

		color_legs = ["#d6f095", "#b4e33d", "#62d92b"]

		boton_push = tk.Button(self, text = "Push", bg = color_push[2], font = ("Open Sans", 15), command = lambda: open_window("push",lista_push, color_push[0], color_push[1], color_push[2]))
		boton_push.pack(expand = True, fill = "both")

		boton_pull = tk.Button(self, text = "Pull", bg = color_pull[2], font = ("Open Sans", 15), command = lambda: open_window("pull",lista_pull, color_pull[0], color_pull[1], color_pull[2]))
		boton_pull.pack(expand = True, fill = "both")

		boton_legs = tk.Button(self, text = "Legs", bg = color_legs[2], font = ("Open Sans", 15), command = lambda: open_window("legs", lista_legs, color_legs[0], color_legs[1], color_legs[2]))
		boton_legs.pack(expand = True, fill = "both")


class ventana_ejercicios(tk.Toplevel):

	def __init__(self, name):

		tk.Toplevel.__init__(self)

		self.title(name.capitalize())
		self.geometry("600x750")
		self.resizable(0,0)

class menu_ejercicios(tk.Frame):

	def __init__(self, parent, lista, name, color1, color2, color3):

		tk.Frame.__init__(self, parent, bg = color1)

		label_texto_cal = tk.Label(self, bg = color1, font = ("Open Sans", 13), text = "Selecciona la fecha del entreno: ")
		label_texto_cal.place(anchor = "center", relx = 0.25, rely = 0.08, width = 245, height = 60)

		self.label_cal = tk.Label(self, bg = color1)
		self.label_cal.place(anchor = "center", relx = 0.75 , rely = 0.08, width = 240, height = 60)

		cal = DateEntry(self.label_cal, width=12, font = ("Open Sans", 15), background='darkblue', foreground='white', borderwidth=2, justify = "center")
		cal.pack(expand = True, fill = "both")
		self.fecha = str(cal.get_date())

		label_texto_sesion = tk.Label(self, bd = 5, bg = color1, font = ("Open Sans", 15), text = "Introduce los datos de la sesión de entrenamiento:")
		label_texto_sesion.place(anchor = "center", relx = 0.5, rely = 0.18)

		self.frame_entreno = tk.Frame(self, bg = color1)
		self.frame_entreno.place(anchor = "center", relx = 0.5, rely = 0.55, width = 550, height = 500)

		for i in range(len(lista)+1):

			self.frame_entreno.rowconfigure(i, weight = 1)

		for i in range(6):

			self.frame_entreno.columnconfigure(i, weight = 1)

		self.frame_entreno.columnconfigure(0, weight = 0)

		self.lista_labels_ejercicios = {}

		for i in range(len(lista)):

			self.lista_labels_ejercicios[i] = tk.Label(self.frame_entreno, text = lista[i], bg = color2, relief = "groove")
			self.lista_labels_ejercicios[i].grid(row = i+1, column = 0, sticky = "news")

		lista_labels_series = {}

		for i in range(5):

			lista_labels_series[i] = tk.Label(self.frame_entreno, text = f"Serie {i+1}", bg = color2, relief = "groove")
			lista_labels_series[i].grid(row = 0, column = i+1, sticky = "news")

		self.lista_labels_entrys = {}
		self.lista_entrys = {}

		for i in range(5*len(lista)):

			self.lista_labels_entrys[i] = tk.Label(self.frame_entreno)
			self.lista_entrys[i] = tk.Entry(self.lista_labels_entrys[i], justify = "center")
			# self.lista_entrys[i].insert(tk.END, "0")
			self.lista_entrys[i].pack(expand = True, fill = "both")

		k = 0
		j = 0
		i = 0

		while k < 5*len(lista):

			for j in range(5):

				self.lista_labels_entrys[k].grid(row = i+1, column = j+1, sticky = "news")

				k+=1

			i+=1

		boton_guardar = tk.Button(self, relief = "groove", text = "Guardar", font = ("Open Sans", 15), bg = color3, command = lambda: self.enviar_datos(lista, name, parent))
		boton_guardar.place(anchor = "center", relx = 0.79, rely = 0.94, width = 200, height = 60)

	def enviar_datos(self, lista, name, ventana):

		self.data = [0,]*len(lista)

		inicio = 0

		fin = 5

		for i in range(len(lista)):

			max = 0.0

			min = 100000

			self.data[i] = [self.fecha, lista[i]]

			for j in range(inicio,fin):

				if self.lista_entrys[j].get().isalpha() == False and self.lista_entrys[j].get()!= "":

					print(self.lista_entrys[j].get())

					self.data[i].append(float(self.lista_entrys[j].get()))

					if max < float(self.lista_entrys[j].get()):

						max = float(self.lista_entrys[j].get())

					if  float(self.lista_entrys[j].get()) > 0 and min > float(self.lista_entrys[j].get()):

						min = float(self.lista_entrys[j].get())
				else:

					self.data[i].append(0.0)


			inicio = fin

			fin += 5

			self.data[i].append(max)

			self.data[i].append(min)


		for i in range(len(lista)):

			self.data[i] = tuple(self.data[i])


		sql = f"INSERT INTO {name} (fecha, ejercicio, serie_1, serie_2, serie_3, serie_4, serie_5, max, min) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
		
		mycursor.executemany(sql, self.data)
	        
		mydb.commit()

		if name == "push":

			ventana.destroy()

		if name == "pull":

			ventana.destroy()

		if name == "legs":

			ventana.destroy()

app = root()

container = menu_principal(app)
container.pack(expand = True, fill = "both")

app.mainloop()