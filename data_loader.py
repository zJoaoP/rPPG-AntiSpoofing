import numpy as np
import threading
import argparse
import time
import cv2
import os


from utils.landmarks import LandmarkPredictor
from utils.extractor import CheeksAndNose


class SpoofInTheWildLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_live = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/live".format(source, split),
			"{0}/{1}/live".format(source, split),
			width=1280,
			height=720,
			points=True
		)

		split_spoof = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/spoof".format(source, split),
			"{0}/{1}/spoof".format(source, split),
			width=1280,
			height=720,
			points=True
		)

		return split_live, split_spoof

	@staticmethod
	def load_train(source):
		return SpoofInTheWildLoader.__load_by_split(source, 'Train')

	@staticmethod
	def load_test(source):
		return SpoofInTheWildLoader.__load_by_split(source, 'Test')

	@staticmethod
	def load_and_store(source, destination):
		train_live, train_spoof = SpoofInTheWildLoader.load_train(source)
		test_live, test_spoof = SpoofInTheWildLoader.load_test(source)

		np.save("{0}/siw_train_live.npy".format(destination), train_live)
		np.save("{0}/siw_train_spoof.npy".format(destination), train_spoof)

		np.save("{0}/siw_test_live.npy".format(destination), test_live)
		np.save("{0}/siw_test_spoof.npy".format(destination), test_spoof)


class OuluLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_full = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}".format(source, split),
			None, #NonAnnotatedDataLoader
		)
		return split_full

	@staticmethod
	def load_train(source):
		return OuluLoader.__load_by_split(source, 'Train_files')

	@staticmethod
	def load_devel(source):
		return OuluLoader.__load_by_split(source, 'Dev_files')

	@staticmethod
	def load_test(source):
		return OuluLoader.__load_by_split(source, 'Test_files')

	@staticmethod
	def load_and_store(source, destination):
		train = OuluLoader.load_train(source)
		devel =  OuluLoader.load_devel(source)
		test = OuluLoader.load_test(source)

		np.save("{0}/oulu_train_full.npy".format(destination), train)
		np.save("{0}/oulu_devel_full.npy".format(destination), devel)
		np.save("{0}/oulu_test_full.npy".format(destination), test)


def get_args():
	parser = argparse.ArgumentParser(description="Código para extração das \
										bases de dados de PAD como SiW e \
										ReplayAttack")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")
	parser.add_argument("time", nargs=None, default=5, action="store", type=int)

	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	if args.source.endswith('SiW_release'):
		if 'siw_train_live.npy' not in os.listdir(args.dest):
			SpoofInTheWildLoader.load_and_store(args.source, args.dest)

		frame_rate = 30
		video_size = frame_rate * args.time
		
		protocol = 2
		if protocol == 1:
			def load_first_protocol(name):
				live = np.load('{0}/siw_{1}_live.npy'.format(args.dest, name))
				spoof = np.load('{0}/siw_{1}_spoof.npy'.format(args.dest, name))
				
				live = live[:, :60]
				spoof = spoof[:, :60]

				part = np.append(spoof, live, axis=0)
				labels = np.append(np.zeros(len(spoof)), np.ones(len(live)))

				np.save('{0}/siw_{1}_x.npy'.format(args.dest, name), part)
				np.save('{0}/siw_{1}_y.npy'.format(args.dest, name), labels)

			load_first_protocol('train')
			load_first_protocol('test')
		elif protocol == 2:
			#Aplicar data augmentation
			#(E não apenas truncar pelo tamanho máximo do vídeo)

			leave_out_medium = 4
			def load_hints(part):
				def parse_name(name):
					result = name.split('-')
					result[-1] = result[-1].split('.')[0]
					return [int(i) for i in result]

				names = list()
				with open(part, 'r') as file:
					for line in file:
						names.append(parse_name(line))

				return names

			train_x, test_x = None, None
			train_y, test_y = list(), list()
			def load_second_protocol(part, label):
				global train_x, test_x
				global train_y, test_y

				items = np.load('{0}/{1}.npy'.format(args.dest, part))
				hints = load_hints('{0}/{1}.txt'.format(args.dest, part))
				for i in range(len(items)):
					item = items[i][:video_size]
					if hints[i][3] == leave_out_medium:
						if test_x is None:
							test_x = item.reshape(1, video_size, 3)
						else:
							test_x = np.append(test_x, np.array([item]), axis=0)

						test_y.append(label)
					else:
						if train_x is None:
							train_x = item.reshape(1, video_size, 3)
						else:
							train_x = np.append(train_x, np.array([item]), axis=0)
						
						train_y.append(label)

			load_second_protocol('siw_train_spoof', 0)
			load_second_protocol('siw_train_live', 1)
			load_second_protocol('siw_test_spoof', 0)
			load_second_protocol('siw_test_live', 1)	

			np.save('{0}/siw_train_x.npy'.format(args.dest), np.array(train_x))
			np.save('{0}/siw_test_x.npy'.format(args.dest), np.array(test_x))
			
			np.save('{0}/siw_train_y.npy'.format(args.dest), np.array(train_y))
			np.save('{0}/siw_test_y.npy'.format(args.dest), np.array(test_y))
	elif args.source.endswith('Oulu_NPU'):
		if 'oulu_train_full.npy' not in os.listdir(args.dest):
			OuluLoader.load_and_store(args.source, args.dest)

		print("TODO")
	else:
		print("Base de dados não suportada.")
