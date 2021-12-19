from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import numpy as np
from PIL import Image, ImageTk


class App():
    def __init__(self, classifier, image_size):
        self.__classifier = classifier
        self.__image_size = image_size
        self.__image_tk = None
        self.__initialize()

    def __initialize(self):
        self.__root = Tk()
        self.__root.title("Boxer CNN")
        self.__root.resizable(False, False)
        self.__master = ttk.Frame(self.__root)
        self.__master.grid(row=0, column=0, padx=8, pady=4)
        file_button = ttk.Button(
            self.__master,
            text="Browse",
            command=self.__load_file
        )
        file_button.grid(row=1, column=0, columnspan=2, sticky=(N, W, E, S))

        self.__canvas = Canvas(
            self.__master, width=self.__image_size, height=self.__image_size)
        self.__canvas.grid(row=0, column=0, columnspan=2, sticky=(N, W, E, S))

    def __load_file(self) -> None:
        file_name = askopenfilename(
            filetypes=[("Image", "*.jpg;*.jpeg;*.png")])
        self.__predict(file_name)

    def __predict(self, file_name):
        try:
            image = Image.open(file_name)
            image_np = np.array(image.resize(
                (self.__image_size, self.__image_size)))
            image_np = np.expand_dims(image_np, axis=0)
            prediction = self.__classifier.predict(image_np)
            print(prediction)

            self.__image_tk = ImageTk.PhotoImage(image)
            image_label = ttk.Label(self.__canvas, image=self.__image_tk)
            image_label.grid(row=0, column=0, sticky=(N, W, E, S))
        except Exception as ex:
            print(ex)
        finally:
            image.close()

    def run(self) -> None:
        self.__root.mainloop()
