from algorithms.default import DefaultStrategy
from algorithms.de_haan import DeHaan

import matplotlib.pyplot as plt
import numpy as np
import cv2

class ModifiedDeHaan(DefaultStrategy):
	def __init__(self, N = 5, M = None):
		M = N if M is None else M
		
		self.N = N
		self.M = M
		
		self.methods = [DeHaan() for i in range(self.N * self.M)]

	def get_min_window(self, frame):
		horizontal_mean = np.mean(frame, axis = (0, 2))
		vertical_mean = np.mean(frame, axis = (1, 2))

		left = [i for i in range(frame.shape[1]) if horizontal_mean[i] > 0][0]
		right = [i for i in range(frame.shape[1] - 1, -1, -1) if horizontal_mean[i] > 0][0]
		
		up = [i for i in range(frame.shape[0]) if vertical_mean[i] > 0][0]
		down = [i for i in range(frame.shape[0] - 1, -1, -1) if vertical_mean[i] > 0][0]
		
		return frame[up : down, left : right, :]

	def get_method(self, i, j):
		return self.methods[i * self.N + j]

	def process(self, frame):
		frame = self.get_min_window(frame)
		offset_m = frame.shape[0] // self.M
		offset_n = frame.shape[1] // self.N
		for i in range(self.N):
			for j in range(self.M):
				left, up = i * offset_n, i * offset_m
				right, down = (i + 1) * offset_n, (i + 1) * offset_m
				
				sub_frame = frame[up : down, left : right, :]

				method = self.get_method(i, j)
				method.process(sub_frame)

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signals = np.array([method.measure_reference(frame_rate = frame_rate, window_size = window_size) for method in self.methods])
		final_signal = np.mean(signals, axis = 0)

		x, y = self.get_fft(final_signal, frame_rate = frame_rate)
		if plot == True:

			plt.subplot(2, 1, 1)
			
			plt.title("DeHaan - Modificado")
			plt.plot(final_signal)

			plt.subplot(2, 1, 2)
			plt.plot(x, y)
			
			plt.show()

		print("[DeHaan Modificado] {0:2f} beats per minute.".format(60.0 * x[np.argmax(y)]))
