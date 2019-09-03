from utils.loaders import GenericDatasetLoader, get_rppg_data
import numpy as np
import argparse
import os


class OuluLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_full = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}".format(source, split),
			None,
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
	parser = argparse.ArgumentParser(description="Code writen to perform the data \
												  extraction task from \
												  'Spoof in the Wild' database.")

	parser.add_argument("source", default=None, action="store")
	parser.add_argument("dest", default=None, action="store")
	parser.add_argument("time", default=5, action="store", type=int)
	# parser.add_argument("protocol", default=1, action="store", type=int)
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	if 'oulu_train_full.npy' not in os.listdir(args.dest):
			OuluLoader.load_and_store(args.source, args.dest)