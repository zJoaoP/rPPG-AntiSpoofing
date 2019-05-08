from utils.extractor import LeftCheek, RightCheek
from algorithms.default import DefaultStrategy
from sklearn.decomposition import FastICA

import matplotlib.pyplot as plt
import numpy as np
import cv2

class ICA(DefaultStrategy):
	def __init__(self):
		self.right_cheek_means = None
		self.left_cheek_means = None

		self.ica = FastICA(whiten = True) #Checar depois.

	def extract_means(self, frame):
		frame[frame == 0.0] = np.nan
		return [np.nanmean(frame[:, :, i]) for i in range(2, -1, -1)]

	def process(self, frame, landmarks):
		frame = np.array(frame, dtype = np.float32)
		
		right_cheek_frame = RightCheek().extract_roi(frame, landmarks)
		left_cheek_frame = LeftCheek().extract_roi(frame, landmarks)
		
		right_cheek_m = self.extract_means(right_cheek_frame)
		left_cheek_m = self.extract_means(left_cheek_frame)
		
		self.right_cheek_means = np.array([right_cheek_m]) if self.right_cheek_means is None else np.append(self.right_cheek_means, [right_cheek_m], axis = 0)
		self.left_cheek_means = np.array([left_cheek_m]) if self.left_cheek_means is None else np.append(self.left_cheek_means, [left_cheek_m], axis = 0)

		print(self.right_cheek_means)

	# http://en.wikipedia.org/wiki/Autocorrelation#Estimation
	def measure_autocorrelation(self, signal):
		mean = signal - np.mean(signal)
		corr = np.correlate(mean, mean, mode = 'full')[-len(signal):]
		result = corr / (np.var(signal) * (np.arange(len(signal), 0, -1)))
		return result

	def select_signal(self, signals, frame_rate = 30, window_size = 45):
		for i in range(len(signals)):
			signals[i] = self.bandpass_filter(signals[i], frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = 64)

		autocorrelations = np.array([self.measure_autocorrelation(signals[i]) for i in range(len(signals))])
		print(autocorrelations)
		return signals[np.argmax(autocorrelations)]

	def measure_reference(self, frame_rate = 30, window_size = 45):
		self.temporal_means = self.temporal_means - np.mean(self.temporal_means, axis = 0)

		signals = self.ica.fit_transform(self.temporal_means)
		periodic_signal = self.select_signal(signals.T, frame_rate = frame_rate, window_size = window_size)
		return signals.T, periodic_signal

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signals, periodic_signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)

		x, y = self.get_fft(periodic_signal, frame_rate = frame_rate)
		if plot == True:
			labels = ['r', 'g', 'b']
			plt.title("ICA")
			plt.subplot(3, 1, 1)
			for i in range(3):
				plt.plot(self.temporal_means.T[i], labels[i])

			plt.subplot(3, 1, 2)
			for i in range(len(signals)):
				plt.plot(signals[i])
			
			plt.subplot(3, 1, 3)
			plt.plot(x, y)

			plt.show()

		print("[ICA] {0:2f} bpm.".format(60.0 * x[np.argmax(y)]))
