import tkinter as tk

from tkcalendar import DateEntry

import mysql.connector

mydb = mysql.connector.connect(host = "localhost", user = "root", password = "x-mania1998", database = "peso")

mycursor = mydb.cursor()

class root(tk.Tk):

	def __init__(self):

		tk.Tk.__init__(self)

		self.title("Peso")
		self.geometry("500x400")
		self.resizable(0,0)

class menu_principal(tk.Frame):

	def __init__(self, parent, var):

		tk.Frame.__init__(self, parent, bg = "#ebc934")

		label_texto_cal = tk.Label(self, bg = "#ebc934", font = ("Open Sans", 13), text = "Selecciona la fecha de hoy: ")
		label_texto_cal.place(anchor = "center", relx = 0.25, rely = 0.2, width = 245, height = 60)

		self.label_cal = tk.Label(self, bg = "#ebc934")
		self.label_cal.place(anchor = "center", relx = 0.75 , rely = 0.2, width = 240, height = 60)

		cal = DateEntry(self.label_cal, width=12, font = ("Open Sans", 15), background='darkblue', foreground='white', borderwidth=2, justify = "center")
		cal.pack(expand = True, fill = "both")

		self.fecha = str(cal.get_date())

		self.label_texto_dia_entreno = tk.Label(self, bg = "#ebc934", font = ("Open Sans", 13), text = "Marca si es día de entreno: ")
		self.label_texto_dia_entreno.place(anchor = "center", relx = 0.25, rely = 0.4, width = 245, height = 80)

		self.label_dia_entreno = tk.Label(self, bg = "#ebc934")
		self.label_dia_entreno.place(anchor = "center", relx = 0.75 , rely = 0.4, width = 245, height = 80)

		checkbox_no = tk.Radiobutton(self.label_dia_entreno, bg = "#ebc934", font = ("Open Sans", 13), text = "No", variable = var, value = "no")
		checkbox_no.pack(anchor = "w")

		checkbox_si = tk.Radiobutton(self.label_dia_entreno, bg = "#ebc934", font = ("Open Sans", 13), text = "Si", variable = var, value = "si")
		checkbox_si.pack(anchor = "w")


		self.label_texto_peso = tk.Label(self, bg = "#ebc934", font = ("Open Sans", 13), text = "Introduce el peso de hoy (kg): ")
		self.label_texto_peso.place(anchor = "center", relx = 0.25 , rely = 0.6, width = 245, height = 60)

		self.label_peso = tk.Label(self, bg = "#ebc934")
		self.label_peso.place(anchor = "center", relx = 0.75 , rely = 0.6, width = 245, height = 60)

		self.peso = tk.Entry(self.label_peso, justify = "center")
		self.peso.pack(expand = True, fill = "both")


		boton_guardar = tk.Button(self, relief = "groove", text = "Guardar", font = ("Open Sans", 15), bg = "#eb7a34", command = lambda: self.enviar_datos(self.fecha, self.peso, var, parent))
		boton_guardar.place(anchor = "center", relx = 0.5, rely = 0.8, width = 200, height = 60)


	def enviar_datos(self, fecha, peso, var, app):

		mi_peso = float(peso.get())

		entreno = var.get()

		self.datos = (fecha, entreno, mi_peso)

		sql = "INSERT INTO mipeso (fecha, entreno, peso) VALUES (%s, %s, %s)"
		
		mycursor.execute(sql, self.datos)
	        
		mydb.commit()

		app.destroy()

app = root()

var = tk.StringVar(value = "no")

container = menu_principal(app, var)
container.pack(expand = True, fill = "both")



app.mainloop()