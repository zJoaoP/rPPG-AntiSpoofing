from utils.loaders import GenericDatasetLoader, get_rppg_data
import numpy as np
import argparse
import os


class SpoofInTheWildLoader:
	@staticmethod
	def __load_by_split(source, split):
		split_live = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/live".format(source, split),
			"{0}/{1}/live".format(source, split),
			border_notation=True
		)

		split_spoof = GenericDatasetLoader.walk_and_load_from(
			"{0}/{1}/spoof".format(source, split),
			"{0}/{1}/spoof".format(source, split),
			border_notation=True
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


def get_args():
	parser = argparse.ArgumentParser(description="Code writen to perform the data \
												  extraction task from \
												  'Spoof in the Wild' database.")

	parser.add_argument("source", default=None, action="store")
	parser.add_argument("dest", default=None, action="store")
	parser.add_argument("time", default=5, action="store", type=int)
	parser.add_argument("protocol", default=1, action="store", type=int)
	return parser.parse_args()


if __name__ == "__main__":
	def load_hints(part):
		def parse_name(name):
			result = name.split('-')
			result[-1] = result[-1].split('.')[0]
			return [int(i) for i in result]

		names = list()
		with open(part, 'r') as file:
			for line in file:
				names.append(parse_name(line))

		return np.array(names)

	args = get_args()
	if 'siw_train_live.npy' not in os.listdir(args.dest):
		SpoofInTheWildLoader.load_and_store(args.source, args.dest)

	frame_rate = 30
	video_size = frame_rate * args.time
	if args.protocol == 0:
		def load_all_data(split):
			live = np.load('{0}/siw_{1}_live.npy'.format(args.dest, split))
			spoof = np.load('{0}/siw_{1}_spoof.npy'.format(args.dest, split))
			
			live = live[:, :video_size]
			spoof = spoof[:, :video_size]

			part = np.append(spoof, live, axis=0)
			labels = np.append(np.zeros(len(spoof)), np.ones(len(live)))

			np.save('{0}/siw_{1}_x.npy'.format(args.dest, split), part)
			np.save('{0}/siw_{1}_y.npy'.format(args.dest, split), labels)

		load_all_data('train')
		load_all_data('test')
	elif args.protocol == 1:
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
	elif args.protocol == 2: # Apply data augmentation.
		leave_out_medium = 3
		def load_second_protocol(part):
			def is_valid_insertion(hint):
				label = 1 if hint[2] == 1 else 0
				attack_type = -1 if label == 1 else hint[2]
				if label == 1:
					return True
				elif (label == 0 and attack_type == 3):
					if part == 'train' and hint[3] != leave_out_medium:
						return True
					elif part == 'test' and hint[3] == leave_out_medium:
						return True
					else:
						return False


			part_x, part_y = None, None
			for category in ['spoof', 'live']:
				data = np.load('{0}/siw_{1}_{2}.npy'.format(args.dest,
															part,
															category))

				hints = load_hints('{0}/siw_{1}_{2}.txt'.format(args.dest,
																part,
																category))
				for i in range(len(data)):
					if is_valid_insertion(hints[i]):
						x = data[i][:video_size]
						y = 1 if hints[i][2] == 1 else 0
						if part_x is None:
							part_x = np.array([x])
							part_y = list([y])
						else:
							part_x = np.append(part_x, np.array([x]), axis=0)
							part_y.append(y)


			print(part_x.shape)
			return part_x, np.array(part_y)

		train_x, train_y = load_second_protocol('train')
		test_x, test_y = load_second_protocol('test')
		
		np.save('{0}/siw_train_x.npy'.format(args.dest), np.array(train_x))
		np.save('{0}/siw_test_x.npy'.format(args.dest), np.array(test_x))
		
		np.save('{0}/siw_train_y.npy'.format(args.dest), np.array(train_y))
		np.save('{0}/siw_test_y.npy'.format(args.dest), np.array(test_y))

	elif args.protocol == 3:
		train_with_print = False
		test_with_print = not train_with_print
		def load_third_protocol(part):
			def is_valid_insertion(hint):
				if hint[2] == 1:
					return True
				elif (part == 'train') and (hint[2] == 2) and train_with_print:
					return True
				elif (part == 'test') and (hint[2] == 2) and test_with_print:
					return True
				elif (part == 'train') and (hint[2] == 3) and test_with_print:
					return True
				elif (part == 'test') and (hint[2] == 3) and train_with_print:
					return True
				else:
					return False


			part_x, part_y = None, None
			for category in ['spoof', 'live']:
				data = np.load('{0}/siw_{1}_{2}.npy'.format(args.dest,
															part,
															category))

				hints = load_hints('{0}/siw_{1}_{2}.txt'.format(args.dest,
																part,
																category))

				for i in range(len(data)):
					if is_valid_insertion(hints[i]):
						x = data[i][:video_size]
						y = 1 if hints[i][2] == 1 else 0

						if part_x is None:
							part_x = np.array([x])
							part_y = list([y])
						else:
							part_x = np.append(part_x, np.array([x]), axis=0)
							part_y.append(y)

			print(part_x.shape)
			return part_x, np.array(part_y)

		train_x, train_y = load_third_protocol('train')
		test_x, test_y = load_third_protocol('test')

		np.save('{0}/siw_train_x.npy'.format(args.dest), np.array(train_x))
		np.save('{0}/siw_test_x.npy'.format(args.dest), np.array(test_x))
		
		np.save('{0}/siw_train_y.npy'.format(args.dest), np.array(train_y))
		np.save('{0}/siw_test_y.npy'.format(args.dest), np.array(test_y))

	train_x = np.load('{0}/siw_train_x.npy'.format(args.dest))
	test_x = np.load('{0}/siw_test_x.npy'.format(args.dest))

	rppg_train_x = get_rppg_data(train_x, frame_rate=frame_rate)
	rppg_test_x = get_rppg_data(test_x, frame_rate=frame_rate)

	np.save('{0}/siw_train_rppg_x.npy'.format(args.dest), rppg_train_x)
	np.save('{0}/siw_test_rppg_x.npy'.format(args.dest), rppg_test_x)
