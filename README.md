# TensorFlow Lite Android Benchmark
![Example image](https://i.imgur.com/S0esLtK.jpg)



Set of tools to benchmark performance of TensorFlow image classification networks on Android devices.

Folder `android/` contains Android app project.
Folder `scripts/` contains Python scripts for converting existing TensorFlow models to TensorFlow Lite models embeddable into the Android application.

Extends official [TensorFlow Android example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) and adds new models and functionality to benchmark performance.

# Generating example networks
Use script `generate.py`

Generates 6 networks pre-trained with ImageNet and saves them to h5 format
- NASNetMobile with input image size 224x224
- EfficientNetB0 with input image size 224x224
- MobileNetV2 with input image size 96x96
- MobileNetV2 with input image size 128x128
- MobileNetV2 with input image size 160x160
- MobileNetV2 with input image size 224x224

# Converting TensorFlow model in h5 format to TensorFlow Lite

## Convert to TFLite format
Use script `convert.py`

Converts TensorFlow model in h5 format to TensorFlow Lite format with optional dynamic range quantization

Usage: `python convert.py -i <input file> [-o <output file>] [-q <quantization level>]`

### Quantization levels:
- `-q 0` - no quantization (default)
- `-q 1` - dynamic range quantiation
- `-q 2` - full integer quantization

## Insert metadata into model
Use script `setMetadata.py`

Inserts metadata into tflite model
Usage: `python setMetadata.py`, then enter information after prompted by terminal

## Verify model metadata (optional)
Use script `getMetadata.py`

Prints metadata from TFLite model file
Usage: `python getMetadata.py -i <input .tflite file>`

## Default naming conventions
`NetworkName_InputSize_QuantizationLevel_HasMetadata`

Example:
1. `generator.py` generates `MobileNetV2_224.h5` - `MobileNetV2` with input size of `224x224`
2. `convert.py` quantizatizes and converts this model to TF Lite format and appends Q followed by quantization level to filename `MobileNetV2_224_Q1.tflite` - quantization level of 1 (see quantization levels above)
3. `setMetadata.py` adds metadata to the model and appends `_M` to filename `MobileNetV2_224_Q1_M.tflite`
