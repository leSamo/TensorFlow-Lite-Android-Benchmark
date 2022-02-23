# generates TF keras models

# NASNetMobile, input resolution (224,224,3)
# EfficientNetB0, input resolution (224,224,3)
# MobileNetV2, input resolutions - (224,224,3), (160,160,3), (128,128,3), (96,96,3)

import tensorflow as tf

NASNetMobile = tf.keras.applications.nasnet.NASNetMobile()
EfficientNetB0 = tf.keras.applications.EfficientNetB0()
MobileNetV2_96 = tf.keras.applications.MobileNetV2(input_shape=(96,96,3))
MobileNetV2_128 = tf.keras.applications.MobileNetV2(input_shape=(128,128,3))
MobileNetV2_160 = tf.keras.applications.MobileNetV2(input_shape=(160,160,3))
MobileNetV2_224 = tf.keras.applications.MobileNetV2()

NASNetMobile.save(f'model/NASNetMobile_224.h5')
EfficientNetB0.save(f'model/EfficientNetB0_224.h5')
MobileNetV2_96.save(f'model/MobileNetV2_96.h5')
MobileNetV2_128.save(f'model/MobileNetV2_128.h5')
MobileNetV2_160.save(f'model/MobileNetV2_160.h5')
MobileNetV2_224.save(f'model/MobileNetV2_224.h5')
