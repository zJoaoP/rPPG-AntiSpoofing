import argparse
import os

from collections import deque


def list_files_in(folder, extension):
	selected_files = deque()
	for root, dirs, files in os.walk(folder):
		for file in files:
			if file.endswith(extension):
				selected_files.append(file)

	return sorted(selected_files)


def get_args():
	parser = argparse.ArgumentParser(description="Código para listar arquivos \
												de acordo com suas extensões.")

	parser.add_argument("source", nargs=None, default=None, action="store")
	parser.add_argument("dest", nargs=None, default=None, action="store")

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	file_names = list_files_in(args.source, '.face')
	with open(args.dest, 'w') as file:
		for name in file_names:
			file.write("{}\n".format(name))
