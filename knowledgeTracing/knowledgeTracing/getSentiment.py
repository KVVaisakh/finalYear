import pickle
import tensorflow_datasets as tfds
import tensorflow as tf

def pad_to_size(vec, size):
	zeros=[0]*(size-len[vec])
	vec.extend[zeros]
	return vec

def sample_predict(sample_pred_text,model,pad):
	dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
	encoder = info.features['text'].encoder

	encoded_sample_pred_text = encoder.encode(sample_pred_text)

	if pad:
		encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text,64)

	encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
	predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

	return (predictions)


def sentiment_result(predictions):
	if predictions > 0.25:
		return 'Positive'
	elif predictions < 0:
		return 'Negative'
	else:
		return 'Neutral'

def predict(message):
	model=tf.keras.models.load_model('home/modelSentimentAnalysis')
	predictions = sample_predict(message, model, pad=False)
	return sentiment_result(predictions)