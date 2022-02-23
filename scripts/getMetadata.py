from tflite_support import metadata
import sys, os, getopt

# parse command line arguments
opts, args = getopt.getopt(sys.argv[1:], "hi:", ["help", "input="])

for option, arg in opts:
	if option == '-h' or option == '--help':
		print("Prints metadata from TFLite model file\nUsage: `python getMetadata.py -i <input .tflite file>`")
		exit()
	elif option == '-i' or option == '--input':
		INPUT_MODEL_PATH = arg
		assert os.path.exists(INPUT_MODEL_PATH), "Input file does not exist"

assert INPUT_MODEL_PATH is not None, "Missing -i option with input model path"

try:
    displayer = metadata.MetadataDisplayer.with_model_file(INPUT_MODEL_PATH)
except ValueError:
	print("No metadata present")
	exit()

print("Metadata present:")
print(displayer.get_metadata_json())

print("Associated file(s) present:")
for file_name in displayer.get_packed_associated_file_list():
    print("file name: ", file_name)
    print("file content:")
    print(displayer.get_associated_file_buffer(file_name))
