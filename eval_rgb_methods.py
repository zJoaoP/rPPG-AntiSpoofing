from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

from utils.wrappers import OpenCV_Wrapper, Kinect_Wrapper
from algorithms.de_haan import DeHaan
from algorithms.green import Green
from algorithms.wang import Wang

from utils.loaders import AsyncVideoLoader

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
		return parser

	def parse_args(self, args = None):
		arguments = self.parser.parse_args(args)
		wrapper = OpenCV_Wrapper(arguments.source) if (not arguments.use_kinect) or arguments.source != 0 else Kinect_Wrapper(width = 1280, height = 720)

		return arguments, wrapper

if __name__ == "__main__":
	args, wrapper = MyArgumentParser().parse_args()
	bgr_predictor = LandmarkPredictor()

	frames_to_skip = 0
	frame_rate = 30

	loader = AsyncVideoLoader(args.source)
	extractor = CheeksAndNose()

	frame_count, frame_limit = 0, args.time * frame_rate
	temporal_means = None
	while frame_count < loader.lenght():
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
				cut_bgr = extractor.extract_roi(cut_bgr, bgr_landmarks).astype(np.float32)
				# for strategy in strategies:
				# 	strategy.process(cut_bgr, bgr_landmarks)

				cv2.imshow('WOW', cut_bgr.astype(np.uint8))

				cut_bgr[cut_bgr == 0.0] = np.nan
				r, g, b = np.nanmean(cut_bgr, axis=(0, 1))
				features = np.array([[r, g, b]])
				if temporal_means is None:
					temporal_means = np.array([[r, g, b]])
				else:
					temporal_means = np.append(temporal_means, np.array([[r, g, b]]), axis=0)

				# for x, y in bgr_landmarks:
				# 	cv2.circle(cut_bgr, (x, y), 1, (0, 255, 0))

				# cv2.imshow("BGR", cut_bgr)

				frame_count += 1

				print("[rPPG Tracker] {0}/{1} frames processados.".format(frame_count, frame_limit))
			else:
				frames_to_skip = args.skip_count	

	# print(temporal_means.shape)
	signal = Wang.extract_rppg(temporal_means)
	fig, (means, rppg) = plt.subplots(2)
	means.plot(temporal_means)
	rppg.plot(signal)

	plt.show()

	print(temporal_means.mean())
	print(temporal_means.max())
	print(temporal_means.min())