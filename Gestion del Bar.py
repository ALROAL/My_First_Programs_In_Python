import tkinter as tk

import mysql.connector

from time import gmtime, strftime, localtime

from PIL import ImageTk, Image


mydb = mysql.connector.connect(host = "localhost", user = "root", password = "x-mania1998", database = "bar")

mycursor = mydb.cursor()

fecha = strftime("%Y-%m-%d | %H:%M:%S", localtime())

class root(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)

        container = tk.Frame(self)
        container.place(relheight = 1, relwidth = 1)

        self.state("zoomed")
        self.resizable(0,0)
        self.title("Gestión de bar")
        self.iconbitmap("logo.ico")

        self.ventana = {}

        self.ventana["mapa_restaurante"] =  mapa_restaurante(container, self)
        self.ventana["mapa_restaurante"].place(relwidth =1, relheight = 1)

        for i in range(9):

            self.ventana[i] =  mesa(container, i+1, self, "#d6c4a3")
            self.ventana[i].place(relwidth = 1, relheight = 1)

        self.show_frame("mapa_restaurante")

    def show_frame(self, cont):

        ventana_mostrada = self.ventana[cont]

        ventana_mostrada.tkraise()

class mapa_restaurante(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent, bg = "#d6c4a3")

        self.bg_image = tk.PhotoImage(file = "planta_restaurante.png")
        self.label = tk.Label(self, image = self.bg_image)
        self.label.image = self.bg_image
        self.label.place(relx = 0.15, rely = 0.1, relwidth = 0.7, relheight = 0.8)

        self.botones_mesas = {}
        self.coords = {1: [175, 365], 2: [340, 265], 3: [455, 205], 4: [375, 485], 5: [540, 385], 6: [655, 320], 7: [575, 600], 
        8: [740, 500], 9: [855,440]}

        for i in range(9):

            self.botones_mesas[i] = tk.Button(self.label, text = "Mesa "+str(i+1), bg = "orange", font = ("Open Sans", 15),  command = lambda i=i: controller.show_frame(i))
            self.botones_mesas[i].place(height = 35, width = 75, x = self.coords[i+1][0], y = self.coords[i+1][1])


class tapa(tk.Button):

    def __init__(self, parent, text, image, command, tipo, ing_2, ing_3):

        tk.Button.__init__(self, parent, fg = "white", font = ("Open Sans", 23, "bold"), compound = "center", 
            relief = "solid", text = text, command = command, image = image)

        self.pack(expand = 1)

        self.tapa = text
        self.tipo = tipo
        self.ing_2 = ing_2
        self.ing_3 = ing_3


class mesa(tk.Frame):

    def __init__(self, parent, num_mesa, controller, color):

        tk.Frame.__init__(self, parent, bg = color)

        self.frame_tapas = tk.Frame(self, bg = "black")
        self.frame_tapas.place(relx = 0.6, rely = 0.15, width = 1099, height = 607, anchor = "n")

        for i in range(3):

            self.frame_tapas.rowconfigure(i, weight = 1)

        for j in range(5):

            self.frame_tapas.columnconfigure(j, weight = 1)

        self.lista_labels_tapas = {}

        for i in range(15):

            self.lista_labels_tapas[i] = tk.Label(self.frame_tapas, bg = "black")

        k = 0
        j = 0
        i = 0

        while k < 15:

            for j in range(5):

                self.lista_labels_tapas[k].grid(row = i, column = j, sticky = "news")
                k+=1

            i+=1

        self.lista_tapas = { 0: ["Hamburguesa", "Carne", "Pan", "Patatas"], 1: ["Lomo", "Carne", "Pan", "Patatas"], 
        2: ["Chipirones", "Pescado", " ", " "], 3: ["Jibia", "Pescado", " ", " "], 4: ["Calamares", "Pescado", " ", " "], 
        5: ["Pinchos","Carne", "Patatas", "Pan"], 6: ["Secreto", "Carne", "Patatas", " "], 7: ["Sardinas", "Pescado", " ", " "], 
        8: ["Boquerones", "Pescado", " ", " "], 9: ["Chorizo", "Carne", "Pan", " "], 10: ["Morcilla", "Carne", "Pan", " "], 
        11: ["Pulpo", "Pescado", "Patatas", " "], 12: ["Alpujarreño", "Carne", "Huevo", "Patatas"], 13: ["Costilla", "Carne", "Patatas", " "], 
        14: ["Tortilla de patatas", "Huevo", "Patatas", " "]}

        self.lista_tapas_pedidas = []

        self.lista_img_tapas = ["burguer.png", "lomo.png", "chipirones.png", "jibia.png", "calamar.png", "pinchos.png", "secreto.png", "sardinas.png",
         "boquerones.png", "chorizo.png", "morcilla.png", "pulpo.png", "alpujarreño.png", "costilla.png", "tortilla.png"]

        self.lista_img_tapas_prep = {}

        for i in range(15):

            self.lista_img_tapas_prep[i] = ImageTk.PhotoImage(Image.open(self.lista_img_tapas[i]).resize((300,200)))


        self.botones_tapas = {}

        self.texto_cuenta = {}

        self.total_tapas_pedidas = 0.0

        self.num_tapas_pedidas = 0.0   

        for i in range(15):

            self.puntos_extra = ""

            if self.lista_tapas[i][0] == "Tortilla de patatas":

                self.texto_cuenta[i] = "Tortilla de patatas......2.50€"

            for j in range(16-len(self.lista_tapas[i][0])):

                self.puntos_extra+="."

                self.texto_cuenta[i] = "Tapa de "+self.lista_tapas[i][0].lower()+self.puntos_extra+" 2.50€\n"

            if self.lista_tapas[i][0] == "Tortilla de patatas":

                self.texto_boton_tapa = "Tortilla\nde patatas"

            else:

                self.texto_boton_tapa = self.lista_tapas[i][0]

            self.botones_tapas[i] = tapa(self.lista_labels_tapas[i], self.texto_boton_tapa, self.lista_img_tapas_prep[i], 
                lambda i=i: self.sumar_tapa("+", i), self.lista_tapas[i][1], self.lista_tapas[i][2], self.lista_tapas[i][3])
            # self.botones_tapas[i].pack(expand = 1)

        self.texto = "Mesa "+str(num_mesa)
        self.label = tk.Label(self, text = self.texto, bg = color, font = ("Verdana", 40, "italic bold"), anchor = "center")
        self.label.place(relx = 0.5 , rely = 0.025, width = 250, height = 60)

        self.label_cuenta = tk.Label(self, text = "Cuenta", bg = color, font = ("Verdana", 20, "italic bold"))
        self.label_cuenta.place(relx = 0.1, rely = 0.1)

        self.tablon = tk.Text(self, undo = True)
        self.tablon.place(relx = 0.05, rely = 0.15, width = 250, height = 610)

        self.boton_deshacer =tk.Button(self.tablon, relief = "groove", text = "Deshacer", font = ("Open Sans", 15), command = lambda: [self.sumar_tapa("-", 0)])
        self.boton_deshacer.place(relx = 0.5, rely = 0.98, anchor = "center")

        self.boton_atras = tk.Button(self, relief = "groove", text = "Atrás", font = ("Open Sans", 20), bg = "#4c9ab0", command = lambda: controller.show_frame("mapa_restaurante"))
        self.boton_atras.place(relx = 0.01, rely = 0.01, width = 200, height = 50)

        self.boton_cocina = tk.Button(self, relief = "groove", text = "Enviar a cocina", font = ("Open Sans", 15), bg = "#d64731", anchor = "center", command = lambda: [self.enviar_cocina(num_mesa), self.sumar_tapa(" ", 0),  controller.show_frame("mapa_restaurante")])
        self.boton_cocina.place(anchor = "se", relx = 0.958, rely = 0.96, width = 200, height = 60)

        self.boton_imprimir_cuenta = tk.Button(self, relief = "groove", text = "Imprimir cuenta", font = ("Open Sans", 20), bg = "#79c79b", command = lambda: self.imprimir_cuenta(num_mesa))
        self.boton_imprimir_cuenta.place(relx = 0.05, rely = 0.90, width = 250, height = 60)


    def sumar_tapa(self, accion, i):

        if accion == "+":

            self.lista_tapas_pedidas.append(self.lista_tapas[i])

            self.tablon.edit_separator()

            self.tablon.insert(tk.END, self.texto_cuenta[i])

            self.num_tapas_pedidas += 1.0

        if accion == "-" and self.num_tapas_pedidas > self.total_tapas_pedidas and self.num_tapas_pedidas > 0:

            self.tablon.edit_undo()

            self.lista_tapas_pedidas.pop()

            self.num_tapas_pedidas -= 1.0

        if accion == " ":

            self.total_tapas_pedidas = self.num_tapas_pedidas


    def enviar_cocina(self, num_mesa):

        if self.num_tapas_pedidas > self.total_tapas_pedidas and self.num_tapas_pedidas > 0:

            self.cuenta = self.tablon.get(self.total_tapas_pedidas + 1, "end -1c")

            print("Para cocina")

            print("Mesa "+str(num_mesa))

            print(self.cuenta)

        print(self.lista_tapas_pedidas)


    def imprimir_cuenta(self, num_mesa):

        self.cuenta = self.tablon.get(1.0, "end -1c")

        self.tablon.delete(1.0, tk.END)

        print("Para cuenta")

        print("Cuenta de mesa "+str(num_mesa))

        print(self.cuenta)

        self.enviar_BBDD()

        self.lista_tapas_pedidas.clear()

        self.num_tapas_pedidas = 0

        self.total_tapas_pedidas = 0


    def enviar_BBDD(self):

        sql = "INSERT INTO tapas (nombre, tipo, ingrediente1, ingrediente2, fecha) VALUES (%s, %s, %s, %s, %s)"

        for i in range(int(self.total_tapas_pedidas)):

            self.lista_tapas_pedidas[i].append(fecha)

        mycursor.executemany(sql, self.lista_tapas_pedidas)
        mydb.commit()

app = root()

app.mainloop()