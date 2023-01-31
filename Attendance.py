import face_recognition
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

path="faces"
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
# git
# config - -
# global user.email
# "you@example.com"
# git
# config - -
# global user.name
# "Your Name"

for cl in myList:
    curimg=cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# random_path=random.choice(myList)
# print(random_path)
#
# # random image path
# random_image_path=path+"/"+random_path
# # read in the image using matplotlib
# image=mpimg.imread(random_image_path)
# # plot the random image
# plt.imshow(image)
# # plt.show()
#
# # identify the face_locations
# imageFaceLoc=face_recognition.face_locations(image)[0]
# # encode the faces
# imageFaceEncode=face_recognition.face_encodings(image)
# # Draw a rectangle to show the location of the faces
# cv2.rectangle(image,(imageFaceLoc[3],imageFaceLoc[0]),(imageFaceLoc[1],imageFaceLoc[2]),(255,0,255),3)
# # cv2.imshow("image",image)
# # cv2.waitKey(0)
#
# print(imageFaceLoc)






def findEncodings(images):
    encodingsList = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodingsList.append(encode)

    return encodingsList

encodeListKnown=findEncodings(images)
encodeListKnown
print("Encoding Complete!!!!!!")

def markAttendance(name):
    with open("attendace.csv","r+")as f:
        myDataList=f.readlines()
        nameList = []
        for line in myDataList:
            entry=line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now=datetime.now()
            dstring=now.strftime("%H-%M-%S")
            f.writelines(f"\n{name},{dstring}")
#
# open the webcam since we want to detect the faces in the webcam
cap=cv2.VideoCapture(0)

while True:
    success,image=cap.read()
    imgS=cv2.resize(image,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgS)
    encodingsCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeface,faceLoc in zip(encodingsCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeface)
        # print(faceDist,matches)

        matchIndex=np.argmin(faceDist)
#
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            # print(name)
        y1,x2,y2,x1=faceLoc
        y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        markAttendance(name)
    cv2.imshow("web",image)
    if cv2.waitKey(1)&0xff==ord("q"):
        break




