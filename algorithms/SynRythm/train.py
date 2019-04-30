import numpy as np

def custom_sin(x, frequency):
	return np.sin(2.0 * np.pi * frequency * x)

def get_fft(y, frame_rate = 30):
	sample_rate = 1.0 / float(frame_rate)
	sample_count = len(y)

	yf = np.fft.fft(y)
	xf = np.linspace(0.0, 1.0 / (2.0 * sample_rate), sample_count // 2)
	return xf, 2.0 / sample_count * np.abs(yf[0 : sample_count // 2])

def gaussian_noise(x, noise_level = 0.0):
	return noise_level * np.random.normal(0, 1, len(x))

def generate_heart_wave(t, main_frequency, noise_level):
	M2 = np.random.random_sample() * 0.4
	M1 = M2 + np.random.random_sample() * 0.5

	phi, theta = [np.random.random_sample() * (2 * np.pi) for i in range(2)]
	
	W1 = main_frequency
	W2 = (5 + np.random.random_sample() * 15) / 60.0
	
	P1 = np.random.random_sample()
	P2 = 1.0 - P1
	
	S = M1 * np.sin((2.0 * np.pi) * W1 * t + phi)
	S += 0.5 * M1 * np.sin((2.0 * np.pi) * 2.0 * W1 * t + phi)
	S += M2 * np.sin((2.0 * np.pi) * W2 * t + theta)
	S += gaussian_noise(t, noise_level = noise_level)
	
	return S

def binary_to_interval(value, min_f = 40.0, max_f = 240.0):
	return (min_f + value * (max_f - min_f)) / 60.0

def interval_to_binary(value, min_f = 40.0, max_f = 240.0):
	return (value - min_f) / (max_f - min_f)

def generate_dataset(size, time, samples_per_second, min_frequency = 40, max_frequency = 240, noise_level = 0.0):
	sample_count = samples_per_second * time
	sample_rate = 1.0 / samples_per_second

	x = np.linspace(0.0, sample_count * sample_rate, sample_count)
	data = np.empty([size, x.shape[0]])
	label = np.zeros(size)
	for i in range(size):
		label[i] = np.random.random_sample()
		data[i] = generate_heart_wave(x, binary_to_interval(label[i], min_f = min_frequency, max_f = max_frequency), noise_level)
		data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i])) # Normalizing between 0 and 1.

	return data, label


from keras.layers import Embedding, LSTM, Dense, Flatten, Reshape
from keras.models import Sequential

def generate_model(input_size, sampling_rate):
	def threshold_accuracy(y_true, y_pred, threshold = 0.5): #threshold (in bpms)
		from keras import backend as K

		true_bpm = binary_to_interval(y_true)
		pred_bpm = binary_to_interval(y_pred)

		return K.mean(K.greater(threshold, K.abs(true_bpm - pred_bpm)))

	model = Sequential()
	model.add(Reshape(target_shape = (input_size // sampling_rate, sampling_rate), input_shape = (input_size, )))
	
	units = [128, 64, 32, 16, 8]
	for u in units:
		model.add(LSTM(u, dropout = 0.3, return_sequences = True))
	
	model.add(Flatten())
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(
		optimizer = "adam",
		loss = "mse",
		metrics = [threshold_accuracy]
	)

	model.summary()

	return model

def data_generator(size, time, samples_per_second, noise_level = 0.0):
	while True:
		yield generate_dataset(size = size, time = time, samples_per_second = samples_per_second, noise_level = noise_level)

if __name__ == "__main__":
	FRAME_RATE = 30
	TIME = 15

	model = generate_model(input_size = TIME * FRAME_RATE, sampling_rate = FRAME_RATE)
	model.fit_generator(generator = data_generator(size = 32, time = TIME, samples_per_second = FRAME_RATE, noise_level = 0.35), steps_per_epoch = 1024, validation_steps = 64, epochs = 100, verbose = 1)