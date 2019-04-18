from utils.wrappers import OpenCV_Wrapper, Kinect_Wrapper
from algorithms.de_haan import DeHaan
import argparse

class MyArgumentParser:
	def __init__(self):
		self.parser = self.build_parser()

	def build_parser(self):
		parser = argparse.ArgumentParser(description = "Detector de Landmarks Faciais")
		parser.add_argument("--source", dest = "source", nargs = "?", default = 0, action = "store",
							help = "Localização do arquivo de origem. (Padrão: %(default)s)")
		parser.add_argument("--skip-count", dest = "skip_count", nargs = "?", default = 60, action = "store", type = int,
							help = "Quantidade de frames a serem saltados caso uma face não seja detectada. (Padrão: %(default)s frames)")
		parser.add_argument("--time-limit", dest = "time_limit", nargs = "?", default = 10, action = "store", type = int,
							help = "Tempo (em segundos) de captura ou análise. (Padrão: %(default)s segundos)")
		parser.add_argument("--use-kinect", dest = "use_kinect", action = "store_true", default = False,
							help = "Define se o programa utilizará (ou não) um kinect para obter as imagens. (Padrão: %(default)s)")
		parser.add_argument("--aproach", dest = "aproach", nargs = "?", default = "DeHaan", action = "store",
							help = "Abordagem no cálculo do rPPG. (Padrão: %(default)s)")
		return parser

	def parse_args(self, args = None):
		arguments = self.parser.parse_args(args)
		strategy = DeHaan() if arguments.aproach == "DeHaan" else None # Buscar forma mais elegante de fazer isto.
		wrapper = OpenCV_Wrapper(arguments.source) if (not arguments.use_kinect) or arguments.source != 0 else Kinect_Wrapper()

		return arguments, wrapper, strategy