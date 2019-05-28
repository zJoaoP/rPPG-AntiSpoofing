class VideoLoader:
	def __init__(self, time = 10, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.time = time

	def is_valid_video(self, filename):
		valid_extensions = ["avi", "mp4", "mov"]
		extension = filename.split('.')[-1]
		return extension in valid_extensions

	def get_video_info(self, filename):
		if not self.is_valid_video(filename):
			return -1, -1
		else:
			stream = cv2.VideoCapture(filename)
			frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_rate = int(stream.get(cv2.CAP_PROP_FPS))
			stream.release()

			return frame_rate, frame_count

	def load_video(self, filename):
		if self.is_valid_video(filename):
			stream = cv2.VideoCapture(filename)
			face_predictor = LandmarkPredictor()
			frame_rate, current_frame = int(stream.get(cv2.CAP_PROP_FPS)), 0
			means = []

			while True:
				success, frame = stream.read()
				if not success:
					break

				face_rect = face_predictor.detect_face(image = frame)
				if face_rect is not None:
					face_left, face_right, face_top, face_bottom = face_rect.left(), face_rect.right(), face_rect.top(), face_rect.bottom()
					cut_face = frame[face_top : face_bottom, face_left : face_right, :]

					landmarks = face_predictor.detect_landmarks(image = cut_face)
					roi = self.roi_extractor.extract_roi(cut_face, landmarks)
					
					if current_frame % 60 == 0 and current_frame > 0:
						print("[VideoLoader] {0} frames extracted from '{1}'.".format(current_frame, filename))
					
					roi = roi.astype(np.float32)
					roi[roi < 5.0] = np.nan

					r = np.nanmean(roi[:, :, 2])
					g = np.nanmean(roi[:, :, 1])
					b = np.nanmean(roi[:, :, 0])

					means += [[r, g, b]]

					current_frame += 1
				else:
					print("[VideoLoader] Skipping '{0}'. Reason: Failure on Acquisition! (It's a folder??)".format(filename))
					return None

			return np.array(means)
		else:
			print("[VideoLoader] Skipping '{0}'. Reason: Invalid Video! (It's a folder??)".format(filename))
			return None

class DataLoader:
	def __init__(self, folder, file_list = None, time = 15, num_channels = 3, roi_extractor = CheeksAndNose()):
		self.roi_extractor = roi_extractor
		self.num_channels = num_channels
		self.time = time

		self.file_list = file_list
		self.folder = folder

		self.loader = VideoLoader(time = time, roi_extractor = roi_extractor)

	def split_raw_data(self, raw_data, split_size, stride = 1):
		data, i = None, 0
		while i + split_size - 1 < len(raw_data):
			print(type(raw_data))
			data = np.array([raw_data[i : i + split_size, :]]) if data is None else np.append(data, np.array([raw_data[i : i + split_size, :]]), axis = 0)
			i += stride

		return data

	# def post_process(self, data, frame_rate = 25):
	# 	min_data_size = min([len(data[i]) for i in range(len(data))])
	# 	min_data_size = min(min_data_size, self.time * frame_rate)
	# 	final_data = None

	# 	for i in range(len(data)):
	# 		sub_videos = self.split_raw_data(np.array(data[i]), split_size = min_data_size, stride = 1)
	# 		final_data = sub_videos if final_data is None else np.append(final_data, sub_videos, axis = 0)

	# 	return final_data

	def load_data(self):
		counter, data = 0, []
		dir_size = len(os.listdir(self.folder))
		for filename in sorted(os.listdir(self.folder)):
			counter += 1
			if (self.file_list is None) or ((self.file_list is not None) and (filename in self.file_list)):
				print("[DataLoader - {0} / {1}] Trying to load video '{2}'.".format(counter, dir_size, filename))
				video_data = self.loader.load_video(filename = self.folder + '/' + filename)
				if video_data is not None:
					data += [video_data]

		return data

class ReplayAttackLoader:
	def custom_shuffle(self, a, b):
		assert len(a) == len(b)
		p = np.random.permutation(len(a))
		return a[p], b[p]

	def load_data(self, folder, time):
		def replace_array_by_cuts(raw_data_bucket, data_loader, split_size):
			new_data = None
			for i in range(len(raw_data_bucket)):
				split = data_loader.split_raw_data(raw_data_bucket[i], split_size, stride = 1)
				new_data = split if new_data is None else np.append(new_data, split, axis = 0)
			
			return new_data

		print("[ReplayAttackLoader] Loading ReplayAttack videos from '{0}/'.".format(folder))
		fake_fixed_loader = DataLoader(folder = folder + "/attack/fixed", time = time)
		fake_hand_loader = DataLoader(folder = folder + "/attack/hand", time = time)
		real_loader = DataLoader(folder = folder + "/real", time = time)

		fake_fixed_data = fake_fixed_loader.load_data()
		fake_hand_data = fake_hand_loader.load_data()
		real = real_loader.load_data()

		min_fixed = min([len(fake_fixed_data[i]) for i in range(len(fake_fixed_data))])
		min_hand = min([len(fake_hand_data[i]) for i in range(len(fake_hand_data))])
		min_real = min([len(real[i]) for i in range(len(real))])

		cut_size = min(min_fixed, min_hand, min_real)

		fake_hand_data = replace_array_by_cuts(np.array(fake_hand_data), fake_hand_loader, cut_size)
		fake_fixed_data = replace_array_by_cuts(np.array(fake_fixed_data), fake_fixed_loader, cut_size)
		real = replace_array_by_cuts(np.array(real), real_loader, cut_size)

		fake = np.append(fake_hand_data, fake_fixed_data, axis = 0)

		labels = np.append(np.zeros(fake.shape[0]), np.ones(real.shape[0]))
		return self.custom_shuffle(np.append(fake, real, axis = 0), to_categorical(labels, 2))