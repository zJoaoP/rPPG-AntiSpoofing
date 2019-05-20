from utils.landmarks import LandmarkPredictor
import numpy as np
import argparse
import cv2

from utils.wrappers import OpenCV_Wrapper, Kinect_Wrapper
from algorithms.de_haan_pos import DeHaanPOS
from algorithms.fast_nir import FastNIR
from algorithms.de_haan import DeHaan
from algorithms.green import Green

class MyArgumentParser:
	def __init__(self):
		self.parser = self.build_parser()

	def build_parser(self):
		parser = argparse.ArgumentParser(description = "Detector de Landmarks Faciais")
		parser.add_argument("--source", dest = "source", nargs = "?", default = 0, action = "store",
							help = "Localização do arquivo de origem. (Padrão: %(default)s)")
		parser.add_argument("--skip-count", dest = "skip_count", nargs = "?", default = 60, action = "store", type = int,
							help = "Quantidade de frames a serem saltados caso uma face não seja detectada. (Padrão: %(default)s frames)")
		parser.add_argument("--time", dest = "time", nargs = "?", default = 10, action = "store", type = int,
							help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
		parser.add_argument("--use-kinect", dest = "use_kinect", action = "store_true", default = False,
							help = "Define se o programa utilizará (ou não) um kinect para obter as imagens. (Padrão: %(default)s)")
		parser.add_argument("--aproach", dest = "aproach", nargs = "?", default = ["Green"], action = "store",
							help = "Abordagem no cálculo do rPPG. (Padrão: %(default)s)")
		return parser

	def parse_strategies(self, aproaches):
		strategies = []
		for aproach in aproaches:
			if aproach == "DeHaan":
				strategies += [DeHaan()]
			elif aproach == "DeHaanPOS":
				strategies += [DeHaanPOS()]
			elif aproach == "Green":
				strategies += [Green()]

		return strategies

	def parse_args(self, args = None):
		arguments = self.parser.parse_args(args)
		strategies = self.parse_strategies(arguments.aproach)
		wrapper = OpenCV_Wrapper(arguments.source) if (not arguments.use_kinect) or arguments.source != 0 else Kinect_Wrapper(width = 1280, height = 720)

		return arguments, wrapper, strategies

if __name__ == "__main__":
	args, wrapper, strategies = MyArgumentParser().parse_args()
	bgr_predictor = LandmarkPredictor()

	frame_rate = wrapper.get_frame_rate()
	frames_to_skip = 0

	frame_count, frame_limit = 0, args.time * frame_rate
	while frame_count < frame_limit:
		(success, bgr, nir) = wrapper.get_frame()
		if not success or cv2.waitKey(1) & 0xFF == ord('q'):
			break
		elif frames_to_skip > 0:
			frames_to_skip -= 1
			continue
		else:
			bgr_rect = bgr_predictor.detect_face(image = bgr)
			if bgr_rect is not None:
				bgr_left, bgr_right, bgr_top, bgr_bottom = bgr_rect.left(), bgr_rect.right(), bgr_rect.top(), bgr_rect.bottom()
				cut_bgr = bgr[bgr_top : bgr_bottom, bgr_left : bgr_right, :]

				bgr_landmarks = bgr_predictor.detect_landmarks(image = cut_bgr)
				for strategy in strategies:
					strategy.process(cut_bgr, bgr_landmarks)

				# for x, y in bgr_landmarks:
				# 	cv2.circle(cut_bgr, (x, y), 1, (0, 255, 0))

				# cv2.imshow("BGR", cut_bgr)

				frame_count += 1

				print("[rPPG Tracker] {0}/{1} frames processados.".format(frame_count, frame_limit))
			else:
				frames_to_skip = args.skip_count	

	for strategy in strategies:
		strategy.show_results(frame_rate = frame_rate, window_size = int(frame_rate * 1.6), plot = True)