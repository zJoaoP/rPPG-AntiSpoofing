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
												  'Oulu-NPU' dataset.")

	parser.add_argument("source", default=None, action="store")
	parser.add_argument("dest", default=None, action="store")
	parser.add_argument("protocol", default=1, action="store", type=int)
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	frame_rate = 30
	if 'oulu_train_full.npy' not in os.listdir(args.dest):
		OuluLoader.load_and_store(args.source, args.dest)

	oulu_train_full = np.load('{0}/oulu_train_full.npy'.format(args.dest))
	oulu_devel_full = np.load('{0}/oulu_devel_full.npy'.format(args.dest))
	oulu_test_full = np.load('{0}/oulu_test_full.npy'.format(args.dest))

	def get_hints(split_file):
		hints = list()
		with open(split_file) as f:
			for line in f:
				hints.append([int(i) for i in line.split('.')[0].split('_')])

		return hints

	if args.protocol == 0:
		def load_all_data(split):
			hints = get_hints('{0}/oulu_{1}_names.txt'.format(args.dest, split))
			x = np.load('{0}/oulu_{1}_full.npy'.format(args.dest, split))[:, :120]
			y = np.zeros(len(x))

			for i in range(len(x)):
				y[i] = 1 if hints[i][3] == 1 else 0

			rppg = get_rppg_data(x, frame_rate=frame_rate)

			np.save('{0}/oulu_{1}_rppg_x.npy'.format(args.dest, split), rppg)
			np.save('{0}/oulu_{1}_x.npy'.format(args.dest, split), x)
			np.save('{0}/oulu_{1}_y.npy'.format(args.dest, split), y)

			print(rppg.shape)

		for split in ['train', 'devel', 'test']:
			load_all_data(split)

	elif args.protocol == 1:
		def load_by_session(split, sessions):
			hints = get_hints('{0}/oulu_{1}_names.txt'.format(args.dest, split))
			x = np.load('{0}/oulu_{1}_full.npy'.format(args.dest, split))
			session_x, session_y = list(), list()
			for i in range(len(x)):
				if hints[i][1] in sessions:
					label = 1 if (hints[i][3] == 1) else 0
					session_x.append(x[i])
					session_y.append(label)
			
			session_x, session_y = np.array(session_x), np.array(session_y)

			split_rppg_x = get_rppg_data(session_x, frame_rate=frame_rate)

			np.save('{0}/oulu_{1}_x.npy'.format(args.dest, split), session_x)
			np.save('{0}/oulu_{1}_y.npy'.format(args.dest, split), session_y)
			np.save('{0}/oulu_{1}_rppg_x.npy'.format(args.dest, split), split_rppg_x)

			print(session_x.shape)

		load_by_session('train', [1, 2])
		load_by_session('devel', [1, 2])
		load_by_session('test', [3])
