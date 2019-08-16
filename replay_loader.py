from utils.loaders import GenericDatasetLoader, slice_and_stride
import numpy as np
import argparse
import os

"""
This class extracts data from Replay Attack dataset.
"""
class ReplayAttackLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_real = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/real".format(source, split),
			"{0}/face-locations/{1}/real".format(source, split)
		)

		split_attack = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/attack".format(source, split),
			"{0}/face-locations/{1}/attack".format(source, split)
		)

		return split_attack, split_real

	@staticmethod
	def load_train(source):
		return ReplayAttackLoader.__load_by_split(source, 'train')

	@staticmethod
	def load_devel(source):
		return ReplayAttackLoader.__load_by_split(source, 'devel')

	@staticmethod
	def load_test(source):
		return ReplayAttackLoader.__load_by_split(source, 'test')

	@staticmethod
	def load_and_store(source, destination):
		destination_files = os.listdir(destination)
		if "rad_train_real.npy" not in destination_files:	
			train_attack, train_real = ReplayAttackLoader.load_train(source)
			np.save("{0}/rad_train_fake.npy".format(destination), train_attack)
			np.save("{0}/rad_train_real.npy".format(destination), train_real)			
		
		if "rad_devel_real.npy" not in destination_files:
			devel_attack, devel_real = ReplayAttackLoader.load_devel(source)
			np.save("{0}/rad_devel_fake.npy".format(destination), devel_attack)
			np.save("{0}/rad_devel_real.npy".format(destination), devel_real)

		if "rad_test_real.npy" not in destination_files:
			test_attack, test_real = ReplayAttackLoader.load_test(source)
			np.save("{0}/rad_test_fake.npy".format(destination), test_attack)
			np.save("{0}/rad_test_real.npy".format(destination), test_real)


def get_args():
	parser = argparse.ArgumentParser(description="Code used to perform the data \
												  extraction task from \
												  Replay Attack database.")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")
	parser.add_argument("time", nargs=None, default=5, action="store", type=int)

	return parser.parse_args()


def slice_partition(name, size, stride, prefix='rad'):
	split_fake = np.load("{0}/rad_{1}_fake.npy".format(args.dest, name))
	split_real = np.load("{0}/rad_{1}_real.npy".format(args.dest, name))
	
	slice_split_fake = slice_and_stride(split_fake, size, stride)
	slice_split_real = slice_and_stride(split_real, size, stride)
	split = np.append(slice_split_fake, slice_split_real, axis=0)
	
	labels_split_fake = np.zeros([len(slice_split_fake)])
	labels_split_real = np.ones([len(slice_split_real)])
	labels = np.append(labels_split_fake, labels_split_real)
	
	np.save("{0}/{1}_{2}_x.npy".format(args.dest, prefix, name), split)
	np.save("{0}/{1}_{2}_y.npy".format(args.dest, prefix, name), labels)


if __name__ == "__main__":
	args = get_args()
	frame_rate = 24
	video_size = frame_rate * args.time
	if 'rad_train_real.npy' not in os.listdir(args.dest):
		ReplayAttackLoader.load_and_store(args.source, args.dest)

	slice_partition('train', video_size, 1)
	slice_partition('devel', video_size, 1)
	slice_partition('test', video_size, 1)