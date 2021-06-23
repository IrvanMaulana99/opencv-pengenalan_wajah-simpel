import cv2
import numpy as np
from PIL import Image
import os

# jalur database gambar wajah
path = 'Dataset'

# mendeteksi wajah berdasar library haarcascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# fungsi untuk mendapat gambar dan labelnya
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # konversi gambar ke mode grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Melatih Dataset Wajah. Mohon Tunggu Sebentar...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# menyimpan konfigurasi model ke file trainer.yml
recognizer.write('Trainer/trainer.yml') 

# tampilkan nomor wajah yang dilatih dan exit
print("\n [INFO] {0} wajah telah terlatih".format(len(np.unique(ids))))