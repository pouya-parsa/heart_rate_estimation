from kivymd.app import MDApp
from kivymd.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.font_definitions import theme_font_styles

from detection import detection

import numpy as np

import seaborn as sns
sns.set()

import cv2

import time

class mainApp(MDApp):

    def build(self):

        
        self.detection = detection()

        container_layout = BoxLayout(orientation="vertical")

        
        
        header_layout = BoxLayout(size_hint=(1, 0.25), padding = (80, 25))
        header_layout.add_widget(
            Image(
                source="hr_image.png", 
                size_hint=(None, None), 
                width=75, 
                height=75,
                pos=(100, 1),))
        
        header_layout.add_widget(
            MDLabel(
                text="Heart Rate via Webcam", 
                halign="center", 
                size_hint=(0.9, 1),
                font_style=theme_font_styles[2]))

        body_layout = BoxLayout(spacing=20, padding=50)

        self.img = Image(size_hint=(0.75, 1))
        
        self.label = MDLabel(
            text="Heart Rate: (Processing)",
            halign="center",
            size_hint=(0.25, 1),
            font_style=theme_font_styles[4]
        )

        body_layout.add_widget(self.img)
        
        body_layout.add_widget(self.label)

        self.capture = cv2.VideoCapture(0)
        
        
        Clock.schedule_interval(self.update, 1.0/38.0)

        container_layout.add_widget(header_layout)
        container_layout.add_widget(body_layout)


        return container_layout

    
    def update(self, dt):

        ret, frame = self.capture.read()

        frame = self.detection.detect_ROI(frame)

        frame = cv2.flip(frame, 0)

        buf = frame.tostring()


        self.label.text = "Heart Rate : (" + str(round(np.mean(np.array(self.detection.bpms)), 1)) + " BPM)"

        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')


        self.img.texture = texture    

if __name__ == '__main__':
    mainApp().run()