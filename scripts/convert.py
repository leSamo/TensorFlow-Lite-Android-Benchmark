# Convert keras TF model to TF lite
# https://www.tensorflow.org/lite/convert

import os, sys, getopt
import tensorflow as tf
import numpy as np

# available quantization levels
NO_QUANTIZATION = 0
DYNAMIC_RANGE_QUANTIZATION = 1
FULL_INTEGER_QUANTIZATION = 2

# parse command line arguments
opts, args = getopt.getopt(sys.argv[1:], "hi:o:q:", ["help", "input=", "output=", "quantize="])

OUTPUT_MODEL_PATH = None
QUANTIZATION_LEVEL = 0

for option, arg in opts:
	if option == '-h' or option == '--help':
		print("Converts TensorFlow model in h5 format to TensorFlow Lite format with optional dynamic range quantization (-q 0 for no quantization (default), -q 1 for dynamic range quantiation, -q 2 for full integer quantization)\nUsage: `python convert.py -i <input file> [-o <output file>] [-q <0,1,2>]`")
		exit()
	elif option == '-i' or option == '--input':
		INPUT_MODEL_PATH = arg
		assert os.path.exists(INPUT_MODEL_PATH), "Input file does not exist"
	elif option == '-o' or option == '--output':
		OUTPUT_MODEL_PATH = arg
	elif option == '-q' or option == '--quantize':
		try:
			QUANTIZATION_LEVEL = int(arg)
			assert QUANTIZATION_LEVEL in [0,1,2]
		except (ValueError, AssertionError):
			print("Quantization level argument is not valid, only values 0 (no quantization), 1 (dynamic range quantization), 2 (full integer quantization) are valid")
			exit()

assert INPUT_MODEL_PATH is not None, "Missing -i option with input model path"

if OUTPUT_MODEL_PATH is None:
	OUTPUT_MODEL_PATH = os.path.splitext(INPUT_MODEL_PATH)[0] + '_Q' + str(QUANTIZATION_LEVEL) + '.tflite'

# load model from file
model = tf.keras.models.load_model(INPUT_MODEL_PATH)

# create a converter to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# set quantization params for converter
if QUANTIZATION_LEVEL == DYNAMIC_RANGE_QUANTIZATION:
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
elif QUANTIZATION_LEVEL == FULL_INTEGER_QUANTIZATION:
	def representative_dataset():
		for _ in range(100):
			data = np.random.rand(*[1 if v is None else v for v in model.input.shape])
			yield [data.astype(np.float32)]
			
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative_dataset

# convert model to TFLite format
tflite_model = converter.convert()

# save model to file
with open(OUTPUT_MODEL_PATH, 'wb') as f:
	f.write(tflite_model)

print("Model saved to", OUTPUT_MODEL_PATH)
