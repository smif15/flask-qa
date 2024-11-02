from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import tf_keras
from io import BytesIO
import requests

#Memuat pre-trained model dan label
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #model pretrained
classifier_model = mobilenet_v2

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt') #unduh label
imagenet_labels = np.array(open(labels_path).read().splitlines()) #merapikan label

#setting input image (224x224px)
IMAGE_SHAPE = (224, 224)

#mendefinisikan model classifier (menambahkan satu layer yg memuat model)
classifier = tf_keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

#fungsi klasifikasi
#input: tautan image dan berapa label yang akan dibuat
def classify(image_path, num_of_classes=5):
    #Preprocessing
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content)).resize(IMAGE_SHAPE)
    #img = Image.open(image_path).resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0

    #Prediction
    result = classifier.predict(img[np.newaxis, ...])[0] #probabilitas dari tiap kelas

    #Mengambil probabilitas tertinggi
    indexes = []
    for i in range(num_of_classes):
        index = np.argmax(result, axis=-1)
        indexes.append(index)
        result[index] = -np.inf
    
    #Return: konversi indeks kelas ke nama kelas dgn probabilitas tertinggi
    return [imagenet_labels[index] for index in indexes]


if __name__ == '__main__':
    print(classify('http://localhost:3000/uploads/2d6debb3-b029-4599-a7a9-a42a486fbffc.png'))