from utils.loaders import GenericDatasetLoader, get_rppg_data
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
		# print(week_content.shape)
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
	parser.add_argument("id", nargs=None, default=None, action="store", type=int)

	return parser.parse_args()

def build_LOOCV(x, y, dest, leave_out_id=0):
	def load_labels():
		def load_names_from_file(file):
			with open(file) as f:
				name_list = [x.split(".")[0] for x in f]
				parsed_names = list()
				for name in name_list:
					parsed_names.append([int(x) for x in name.split('_')[:3]])

				return parsed_names

		names = list()
		for i in range(3):
			f = load_names_from_file("{0}/3DMAD_week{1}.txt".format(dest, i + 1))
			names += f

		return names

	labels = load_labels()
	train_x, train_y = None, None
	test_x, test_y = None, None
	for i in range(len(x)):
		if labels[i][0] == leave_out_id:
			if test_x is None:
				test_x = np.array([x[i]])
				test_y = np.array([y[i]])
			else:
				test_x = np.append(test_x, np.array([x[i]]), axis=0)
				test_y = np.append(test_y, np.array([y[i]]), axis=0)
		else:
			if train_x is None:
				train_x = np.array([x[i]])
				train_y = np.array([y[i]])
			else:
				train_x = np.append(train_x, np.array([x[i]]), axis=0)
				train_y = np.append(train_y, np.array([y[i]]), axis=0)
	
	train_rppg_x = get_rppg_data(train_x, frame_rate=30)
	test_rppg_x = get_rppg_data(test_x, frame_rate=30)

	np.save("{0}/3DMAD_train_rppg_x.npy".format(dest), train_rppg_x)
	np.save("{0}/3DMAD_test_rppg_x.npy".format(dest), test_rppg_x)

	np.save("{0}/3DMAD_train_x.npy".format(dest), train_x)
	np.save("{0}/3DMAD_test_x.npy".format(dest), test_x)

	np.save("{0}/3DMAD_train_y.npy".format(dest), train_y)
	np.save("{0}/3DMAD_test_y.npy".format(dest), test_y)

if __name__ == "__main__":
	args = get_args()
	frame_rate = 30
	video_size = frame_rate * args.time
	if '3DMAD_week1.npy' not in os.listdir(args.dest):
		Loader3DMAD.load_and_store(args.source, args.dest)

	week1 = np.load("{0}/3DMAD_week1.npy".format(args.dest))
	week2 = np.load("{0}/3DMAD_week2.npy".format(args.dest))
	week3 = np.load("{0}/3DMAD_week3.npy".format(args.dest))

	train_x = week1
	train_x = np.append(train_x, week2, axis=0)
	train_x = np.append(train_x, week3, axis=0)
	
	train_y = np.append(np.ones(len(week1) + len(week2)), np.zeros(len(week3)))
	# build_LOOCV(train_x, train_y, args.dest, args.id)
	x = get_rppg_data(train_x, frame_rate=30)

	np.save('./data/3DMAD_rppg_x.npy', x)
	np.save('./data/3DMAD_y.npy', train_y)
	# slice_partition('train', video_size, 1)
	# slice_partition('devel', video_size, 1)
	# slice_partition('test', video_size, 1)