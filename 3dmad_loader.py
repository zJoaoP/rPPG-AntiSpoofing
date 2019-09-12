from utils.loaders import GenericDatasetLoader, slice_and_stride
import numpy as np
import argparse
import os

"""
This class extracts data from Replay Attack dataset.
"""
class Loader3DMAD:
	@staticmethod
	def __load_by_split(source, split):
		week_content = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}".format(source, split),
			None
		)
		print(week_content.shape)
		# split_real = GenericDatasetLoader.walk_and_load_from(
		# 	"{0}/{1}/real".format(source, split),
		# 	"{0}/face-locations/{1}/real".format(source, split)
		# )

		# split_attack = GenericDatasetLoader.walk_and_load_from(
		# 	"{0}/{1}/attack".format(source, split),
		# 	"{0}/face-locations/{1}/attack".format(source, split)
		# )
		return week_content

	@staticmethod
	def load_week(source, week_id):
		return Loader3DMAD.__load_by_split(source, 'Data[week{0}]'.format(week_id))

	@staticmethod
	def load_and_store(source, destination):
		destination_files = os.listdir(destination)
		for i in range(3):
			week_name = "3DMAD_week{0}.npy".format(i + 1)
			if week_name not in destination_files:
				week_content = Loader3DMAD.load_week(source, i + 1)
				np.save("{0}/{1}".format(destination, week_name), week_content)


def get_args():
	parser = argparse.ArgumentParser(description="Code used to perform the data \
												  extraction task from \
												  3DMAD database.")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")
	parser.add_argument("time", nargs=None, default=1, action="store", type=int)

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
	frame_rate = 30
	video_size = frame_rate * args.time
	if '3DMAD_week1.npy' not in os.listdir(args.dest):
		Loader3DMAD.load_and_store(args.source, args.dest)

	# slice_partition('train', video_size, 1)
	# slice_partition('devel', video_size, 1)
	# slice_partition('test', video_size, 1)