from kivymd.app import MDApp
from kivymd.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivymd.uix.label import MDLabel
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import time

import cv2

import time

class mainApp(MDApp):

    def build(self):
        
        self.data_buffer = []
        self.times = [0.4573941230773926, 0.5243048667907715, 0.5884156227111816, 0.6598880290985107, 0.7261257171630859, 0.7935934066772461, 0.8618490695953369, 0.9263031482696533, 0.9928390979766846, 1.0607123374938965, 1.1237776279449463, 1.1914820671081543, 1.2601053714752197, 1.3237175941467285, 1.3914439678192139, 1.4603352546691895, 1.5236155986785889, 1.5936810970306396, 1.6621437072753906, 1.7242372035980225, 1.7914845943450928, 1.8629605770111084, 1.9265611171722412, 1.9942429065704346, 2.0627574920654297, 2.1286895275115967, 2.192129135131836, 2.2594246864318848, 2.326779842376709, 2.397487163543701, 2.460686206817627, 2.5290303230285645, 2.602142095565796, 2.6607887744903564, 2.727782964706421, 2.797205924987793, 2.869267463684082, 2.9277684688568115, 2.9965527057647705, 3.063271999359131, 3.1276726722717285, 3.1966631412506104, 3.2638580799102783, 3.3359742164611816, 3.3945250511169434, 3.4696097373962402, 3.531897783279419, 3.596130609512329, 3.664821147918701, 3.7321441173553467, 3.794802665710449, 3.86594557762146, 3.9348957538604736, 4.000079393386841, 4.067004680633545, 4.13087010383606, 4.199794769287109, 4.26450777053833, 4.331776857376099, 4.39981746673584, 4.470014572143555, 4.531943082809448, 4.600418567657471, 4.667673826217651, 4.731394529342651, 4.8003249168396, 4.8690619468688965, 4.937388181686401, 4.9999401569366455, 5.067350387573242, 5.135463237762451, 5.199345588684082, 5.267485857009888, 5.336625337600708, 5.409091234207153, 5.469179630279541, 5.538147211074829, 5.602144718170166, 5.667215824127197, 5.73421573638916, 5.803366184234619, 5.872560262680054, 5.936913728713989, 6.005550146102905, 6.071615934371948, 6.135164737701416, 6.2039923667907715, 6.274175643920898, 6.335222959518433, 6.4032487869262695, 6.474201202392578, 6.539541482925415, 6.602487564086914, 6.671722173690796, 6.73976993560791, 6.803497314453125, 6.872542381286621, 6.941556453704834, 7.008325815200806, 7.072098255157471, 7.139697313308716, 7.206878900527954, 7.271030426025391, 7.339463472366333, 7.40746808052063, 7.475736379623413, 7.539863586425781, 7.608581304550171, 7.675726652145386, 7.739495754241943, 7.8068687915802, 7.876708030700684, 7.938722133636475, 8.008517503738403, 8.086267709732056, 8.143656492233276, 8.208364248275757, 8.278966426849365, 8.344727516174316, 8.407656192779541, 8.475274085998535, 8.542881727218628, 8.611860036849976, 8.674273490905762, 8.743529796600342, 8.811385154724121, 8.875312089920044, 8.943866491317749]
        self.bpms = []
        self.buffer_size = 128
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

        layout = BoxLayout()
        
        self.img = Image()
        
        self.label = MDLabel(
            text="Heart Rate: (Processing)",
            halign="center",
        )

        layout.add_widget(self.img)
        
        layout.add_widget(self.label)

        time.sleep(1) 
        self.capture = cv2.VideoCapture(0)
        
        
        Clock.schedule_interval(self.update, 1.0/38.0)

        return layout

    
    def update(self, dt):

        ret, frame = self.capture.read()

        frame = self.detect_ROI(frame)


        frame = cv2.flip(frame, 0)

        buf = frame.tostring()


        self.label.text = "Heart Rate : (" + str(round(np.mean(np.array(self.bpms)), 2)) + "!)"

        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')


        self.img.texture = texture
    
    
    def detect_ROI(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = self.detector(gray, 0)
        
        if len(rects) > 0:
            
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])

            if y < 0:
                return 

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_frame = frame[y: y + h, x: x + w]

            grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            shape = self.predictor(grayf, rects[0])
            shape = face_utils.shape_to_np(shape)
            
            cv2.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                            (shape[12][0],shape[33][1]), (0,50,50), 0)
            
            cv2.rectangle(frame, (shape[4][0], shape[29][1]), 
                            (shape[48][0],shape[33][1]), (0,255,0), 0)

            
            ROI1 = frame[shape[29][1]:shape[33][1], shape[54][0]: shape[12][0]] # right chin

            mean = ROI1.mean(axis=(0, 1))

            g = mean[1]
        
            if (abs(mean[1] - np.mean(self.data_buffer)) > 10) and (len(self.data_buffer) > 99):
                g = self.data_buffer[-1]
            
            self.data_buffer.append(g)

            if len(self.data_buffer) > 128 :
            
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                
    #             times = times[-buffer_size:]
                
    #             times[1:] = times[1:] - times[0]
                
                self.process()
        
        else:
            cv2.putText(frame, "No FACE DETECTED", (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, (0, 256, 256))

        return frame

    
    # implementing Realt Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam H.Rahman, M.U Ahmed

    def process(self):
        
        data_buffer = np.array(self.data_buffer)

        """
        Detrending
        Remove unwanted trend from series
        the collected RGB signals will be drfting and noising
        
        """
        data_buffer = signal.detrend(data_buffer, axis=0)
        
        
        # Filtering
        filter_ = np.hamming(128) * 1.4 + 0.6
    #     filter_ = filter_.reshape(128, 1)
        x_filtered = filter_ * data_buffer
        
        # Normalization
    #     data_buffer_normalized = (x_filtered - x_filtered.mean()) \
    #                                     / x_filtered.std()
        
        data_buffer_normalized = x_filtered / np.linalg.norm(x_filtered)
        
        fft = np.fft.fft(data_buffer_normalized * 10)
        fft = np.abs(fft) ** 2
        
        times_ = np.array(self.times)
        
        selected_freq = (times_ > 0.75) & (times_ < 3)
        times_ = times_[selected_freq]
        
    #     plt.plot(times_, fft[selected_freq][:, 0])
        
        bpm = len(signal.find_peaks(fft[selected_freq][:])[0]) / (self.times[-1] - self.times[0]) * 60
        
        self.bpms.append(bpm)

if __name__ == '__main__':
    mainApp().run()