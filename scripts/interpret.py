from multiprocessing.sharedctypes import Value
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import getopt
import os
import sys

# TODO: Support any resolution, not only 224x224

sampleCount = 1000;

# parse command line arguments
opts, args = getopt.getopt(sys.argv[1:], "hm:s:", ["help", "model=", "samples="])

for option, arg in opts:
	if option == '-h' or option == '--help':
		print("Evaluates TFLite image classification model\nUsage: `python interpret.py -m <path to model> [-s <samples count>]`")
		exit()
	elif option == '-m' or option == '--model':
		MODEL_PATH = arg
		assert os.path.exists(MODEL_PATH), "Model file does not exist"
	elif option == '-s' or option == '--samples':
		try:
			sampleCount = int(arg)
			assert sampleCount > 0 and sampleCount <= 10000
		except (ValueError, AssertionError):
			print("Sample count (-s parameter) should be a valid positive integer <= 10000")
			exit()

assert MODEL_PATH is not None, "Missing -m option with path to model to evaluate"

ds = tfds.load('imagenet_v2')

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

top1 = 0
top5 = 0

for sampleIndex, sample in enumerate(list(ds['test'])):
	if (sampleIndex == sampleCount):
		break;

	img = sample['image']
	ratio = 225.0 / min(np.shape(img)[0:2])
	resized = tf.image.resize(img, [int(np.shape(img)[0] * ratio), int(np.shape(img)[1] * ratio)], preserve_aspect_ratio=True)
	cropped = tf.image.random_crop(resized, (224,224,3))
	preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(cropped)

	interpreter.set_tensor(input_details[0]['index'], [preprocessed])
	interpreter.invoke()

	output_data = interpreter.get_tensor(output_details[0]['index'])
	top5indices = (-output_data[0]).argsort()[:5]
	expectedLabel = int(sample['label'])

	if expectedLabel in top5indices:
		top5 += 1

		if top5indices[0] == expectedLabel:
			top1 += 1

	print(f"Progress: {sampleIndex + 1}/{sampleCount}", end="\r")

	"""
	preds = tf.keras.applications.mobilenet_v2.decode_predictions(output_data, top=5)

	for (index, (actualId, name, prob)) in enumerate(preds[0]):
		print(name, prob)
	"""

print("\n\n---------- RESULTS ----------")
print("Model:", MODEL_PATH)
print(f"Top1 accuracy: {100.0 * top1 / sampleCount}% ({top1})")
print(f"Top5 accuracy: {100.0 * top5 / sampleCount}% ({top5})")
print("Sample images count:", sampleCount)
