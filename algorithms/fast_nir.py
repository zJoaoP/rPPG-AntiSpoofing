from utils.extractor import CheeksOnly, FaceWithoutEyes, CheeksAndNose

from algorithms.default import DefaultStrategy
import matplotlib.pyplot as plt
import numpy as np
import cv2

class FastNIR(DefaultStrategy):
	def __init__(self, temporal_means = None):
		self.temporal_means = temporal_means

	def process(self, frame, landmarks):
		frame = CheeksAndNose().extract_roi(frame, landmarks)
		frame = np.array(frame, dtype = np.float32)
		frame[frame == 0.0] = np.nan

		if self.temporal_means is None:
			self.temporal_means = np.array([np.nanmean(frame)])
		else:
			self.temporal_means = np.append(self.temporal_means, np.nanmean(frame))

	def measure_reference(self, frame_rate = 30, window_size = 45):
		def get_order(size):
			return (size - 6) // 3

		signal = self.bandpass_filter(self.temporal_means, frame_rate = frame_rate, min_freq = 0.6, max_freq = 4.0, order = get_order(len(self.temporal_means)))
		return signal - np.mean(signal)

	def show_results(self, frame_rate = 30, window_size = 60, plot = True):
		signal = self.measure_reference(frame_rate = frame_rate, window_size = window_size)
		x, y = self.get_fft(signal, frame_rate = frame_rate)
		if plot:
			plt.title("NIR")

			plt.subplot(2, 1, 1)
			plt.plot(signal)

			plt.subplot(2, 1, 2)
			plt.plot(x, y)
			
			plt.show()

		print("[NIR] {0:2f} bpm.".format(60.0 * x[np.argmax(y)]))
