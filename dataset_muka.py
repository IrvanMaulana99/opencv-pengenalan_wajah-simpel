import cv2
import os

# kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640) # atur lebar video
cam.set(4, 480) # atur tinggi video

# menggunakan haarcascade untuk deteksi muka
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# id muka
face_id = input('\n enter user id')

print("\n [INFO] Inisialiasi Capture Wajah....")
# inisialisasi hitungan
count = 0

while(True):

    ret, img = cam.read()
    # menerapkan filter grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # menampilkan square di area wajah
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # simpan wajah yang di capture ke folder sebagai dataset - format gambar jpg
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30: 
        break

# Penutupan Program
print("\n [INFO] Menutup Program")
cam.release()
cv2.destroyAllWindows()