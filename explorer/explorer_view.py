import sys
sys.path.append("..")

import sys
import time
import random
import tkinter as tk 

import numpy as np
import imageio
from PIL import Image, ImageTk, ImageDraw

import util

from explorer_model import ExplorerModel

class ExplorerView(tk.Frame):
    def __init__(self, opt, model: ExplorerModel, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.name = 'Deep Voxels Explorer'
        self.opt = opt
        self.canvas_width = 502
        self.canvas_height = 502
        self.model = model
        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)

        self._init_sliders()
        self._init_image_view()

    def _init_image_view(self):
        self.image_view = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.image_data = None
        self.image_view.pack()
        init_image = imageio.imread('./background.png')
        init_image = Image.fromarray(init_image)
        init_image = init_image.resize((self.canvas_width, self.canvas_height), Image.ANTIALIAS)
        self.image_data = ImageTk.PhotoImage(init_image)
        self.image_view.create_image(0, 0, anchor="nw", image=self.image_data)

    def _init_sliders(self):
        sliders = {'X': self.create_set_translation_fun(0),
                    'Y': self.create_set_translation_fun(1),
                    'Z': self.create_set_translation_fun(2),
                    'Roll': self.create_set_rotation_fun(0),
                    'Pitch': self.create_set_rotation_fun(1),
                    'Yaw': self.create_set_rotation_fun(2),
                    }
        self.sliders = []
        for key in sliders:
            slider = tk.Scale(self, from_=-2.0, to=2.0, resolution=0.01, orient='horizontal', command=sliders[key], label=key)
            self.sliders.append(slider)
            self.sliders[-1].pack()

    def update_view(self):
        pose = [self.translation, self.rotation]
        image, _ = self.model.request_image(pose)
        image *= 2 ** 8 - 1
        image = image.round().clip(0, 2 ** 8 - 1)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        self.image_data = ImageTk.PhotoImage(image)
        self.image_view.create_image(0, 0, anchor="nw", image=self.image_data)


    def create_set_translation_fun(self, id):
        def set_translation(x):
            self.translation[id] = float(x)
            print(self.translation)
            self.update_view()
        return set_translation


    def create_set_rotation_fun(self, id):
        def set_rotation(x):
            self.rotation[id] = float(x)
            print(self.rotation)
            self.update_view()
        return set_rotation



# if __name__ == "__main__":
#     root = tk.Tk()
#     ExplorerView(0, 0, root).pack(side="top", fill="both", expand=True)
#     root.mainloop()
