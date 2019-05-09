from utils.extractor import LeftCheek, RightCheek

from algorithms.default import DefaultStrategy
from algorithms.de_haan import DeHaan
import matplotlib.pyplot as plt
import numpy as np
import cv2


class TestMaups(DefaultStrategy):
	def __init__(self):
		self.right_cheek_means = None
		self.left_cheek_means = None

	def process(self, frame, landmarks):
		frame = np.array(frame, dtype = np.float32)
		
		right_cheek_frame = RightCheek().extract_roi(frame, landmarks)
		left_cheek_frame = LeftCheek().extract_roi(frame, landmarks)
		
		right_cheek_m = self.extract_means(right_cheek_frame)
		left_cheek_m = self.extract_means(left_cheek_frame)
		
		self.right_cheek_means = np.array([right_cheek_m]) if self.right_cheek_means is None else np.append(self.right_cheek_means, [right_cheek_m], axis = 0)
		self.left_cheek_means = np.array([left_cheek_m]) if self.left_cheek_means is None else np.append(self.left_cheek_means, [left_cheek_m], axis = 0)

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		right_signal = DeHaan(self.right_cheek_means).measure_reference(frame_rate = frame_rate, window_size = window_size)
		left_signal = DeHaan(self.left_cheek_means).measure_reference(frame_rate = frame_rate, window_size = window_size)
		if plot == True:			
			plt.title("TestMaups")
			plt.plot(right_signal, 'k')
			plt.plot(left_signal, 'b')
	
			plt.show()