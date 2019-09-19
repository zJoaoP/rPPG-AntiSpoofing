import keras.backend as K
import tensorflow as tf


def FAR(y_true, y_pred):
	y_true = K.argmax(y_true)
	y_pred = K.argmax(y_pred)
	return K.mean(K.all([1 - y_true, y_pred], axis=0))

def FRR(y_true, y_pred):
	y_true = K.argmax(y_true)
	y_pred = K.argmax(y_pred)
	return K.mean(K.all([1 - y_pred, y_true], axis=0))

def HTER(y_true, y_pred):
	far = FAR(y_true, y_pred)
	frr = FRR(y_true, y_pred)
	return (far + frr) / 2.0

def APCER(y_true, y_pred):
	y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
	y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
	count_fake = K.sum(1 - y_true)

	false_positives = K.all([K.equal(y_true, 0), K.equal(y_pred, 1)], axis=0)
	false_positives = K.sum(K.cast(false_positives, dtype='float32'))
	return K.switch(K.equal(count_fake, 0), 0.0, false_positives / count_fake)

def BPCER(y_true, y_pred):
	y_true = K.cast(K.argmax(y_true, axis=1), dtype='float32')
	y_pred = K.cast(K.argmax(y_pred, axis=1), dtype='float32')
	count_real = K.sum(y_true)

	false_negatives = K.all([K.equal(y_true, 1), K.equal(y_pred, 0)], axis=0)
	false_negatives = K.sum(K.cast(false_negatives, dtype='float32'))
	return K.switch(K.equal(count_real, 0), 0.0, false_negatives / count_real)

def ACER(y_true, y_pred):
	apcer = APCER(y_true, y_pred)
	bpcer = BPCER(y_true, y_pred)

	return (apcer + bpcer) / 2

# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def AUC_ROC(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	K.get_session().run(tf.local_variables_initializer())
	return auc