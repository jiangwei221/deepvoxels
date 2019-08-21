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
        self.winfo_toplevel().title(self.name)
        self.opt = opt
        self.canvas_width = 502
        self.canvas_height = 502
        self.model = model
        self._translation = np.zeros(3)
        self._rotation = np.zeros(3)

        self._init_sliders()
        self._init_image_view()
        self._init_buttons()
        self.update_view()
        self.bind_all('<Key>', self.key_pressed)

    @property
    def translation(self):
        return self._translation.copy()
    
    @translation.setter
    def translation(self, value):
        self._translation = value
        if self.sliders['X'] != self._translation[0]:
            self.sliders['X'].set(self._translation[0])
        if self.sliders['Y'] != self._translation[1]:
            self.sliders['Y'].set(self._translation[1])
        if self.sliders['Z'] != self._translation[2]:
            self.sliders['Z'].set(self._translation[2])
        self.update_view()

    @property
    def rotation(self):
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        if self.sliders['Pitch'] != self._rotation[0]:
            self.sliders['Pitch'].set(self._rotation[0])
        if self.sliders['Yaw'] != self._rotation[1]:
            self.sliders['Yaw'].set(self._rotation[1])
        if self.sliders['Roll'] != self._rotation[2]:
            self.sliders['Roll'].set(self._rotation[2])
        self.update_view()

    def key_pressed(self, event):
        if event.char in ['w', 'a', 's', 'd']:
            self.update_translation_by_key_pressed(event.char)

    def update_translation_by_key_pressed(self, char):
        # Update by keyboard input may cause lag
        update_resolution = 0.05
        translation = self.translation
        if char == 'w':
            translation[2] += update_resolution
        elif char == 's':
            translation[2] -= update_resolution
        elif char == 's':
            translation[2] += update_resolution
        elif char == 'a':
            translation[0] -= update_resolution
        elif char == 'd':
            translation[0] += update_resolution
        self.translation = translation

    def _init_buttons(self):
        self.clear_button = tk.Button(self, text="Reset values",
                                        command=self.reset_values)
        self.clear_button.pack()

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
        sliders = {
                    'X': [self.create_set_translation_fun(0), -10, 10, 0.1],
                    'Y': [self.create_set_translation_fun(1), -10, 10, 0.1],
                    'Z': [self.create_set_translation_fun(2), -10, 10, 0.1],
                    'Pitch': [self.create_set_rotation_fun(0), -np.pi, np.pi, 0.01],
                    'Yaw': [self.create_set_rotation_fun(1), -np.pi, np.pi, 0.01],
                    'Roll': [self.create_set_rotation_fun(2), -np.pi, np.pi, 0.01],
                    }
        self.sliders = {}
        for key in sliders:
            slider = tk.Scale(self, from_=sliders[key][1],
                                    to=sliders[key][2],
                                    resolution=sliders[key][3],
                                    orient='horizontal',
                                    command=sliders[key][0],
                                    label=key)
            self.sliders[key] = slider
            self.sliders[key].pack()

    def update_view(self):
        pose = [self.translation, self.rotation]
        try:
            image, _ = self.model.request_image(pose)
            image *= 255
            image = image.round().clip(0, 255)
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            self.image_data = ImageTk.PhotoImage(image)
            self.image_view.create_image(0, 0, anchor="nw", image=self.image_data)
        except ValueError:
            pass

    def create_set_translation_fun(self, id):
        def set_translation(x):
            translation = self.translation
            translation[id] = float(x)
            self.translation = translation
        return set_translation

    def create_set_rotation_fun(self, id):
        def set_rotation(x):
            rotation = self.rotation
            rotation[id] = float(x)
            self.rotation = rotation
        return set_rotation

    def reset_values(self):
        self.translation = np.zeros(3)
        self.rotation = np.zeros(3)
