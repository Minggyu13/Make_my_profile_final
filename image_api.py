import cv2
import base64
import os, sys
from PIL import Image
from tkinter import *
import subprocess


class ImageApi:


    def __init__(self, file_path):
        self.img = cv2.imread(file_path)
        self.file_path = file_path.split('/')
        self.file_name = self.file_path[5].split('.')[0] # file name of filepath
        self.linux_file_path = f'mnt/c/{self.file_path[1]}/{self.file_path[2]}/{self.file_path[3]}/{self.file_path[4]}/{self.file_path[5]}' # We are using the Ubuntu terminal, so we need to define 'window_file_path' as 'linux_file_path'

        command = f"wsl python3 /mnt/c/Users/qa762/Desktop/Make_my_profile/yolov5/detect.py --save-txt --save-conf --name {self.file_name} --weights /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/runs/train/Profile_yolov5s_results/weights/best.pt --conf 0.3 --source /{self.linux_file_path}"
        subprocess.run(command, shell=True)



    def show_image(self):

        result_img = self.img.copy()
        box = list()
        label_file_path = f'C:/Users/qa762/Desktop/Make_my_profile/yolov5/runs/detect/{self.file_name}/labels/{self.file_name}.txt'
        with open(label_file_path, 'r') as file:
            # 파일에서 텍스트 읽기
            text = file.read().strip()

        box = text.split()
        print("bbox list:", box[0:-2])
        print("conf:", box[-1])

        yolov5_xmin = max(int((float(box[1]) - float(box[3]) / 2) * self.img.shape[1]),0)
        yolov5_ymin = max(int((float(box[2]) - float(box[4]) / 2) * self.img.shape[0]),0)
        yolov5_width = int(float(box[3]) * self.img.shape[1])
        yolov5_height = int(float(box[4]) * self.img.shape[0])


        face_img = self.img[yolov5_ymin:yolov5_ymin + yolov5_height, yolov5_xmin:yolov5_xmin + yolov5_width]

        # else:
        #     print('얼굴이 검출되지 않았습니다!')


        cv2.imshow('Window Name', face_img)
        cv2.waitKey(0)  # 키 이벤트를 기다림, 0은 무한 대기
        cv2.destroyAllWindows()  # 열린 창을 모두 닫음

    def save_profile(self):

        file_path = f'C:/Users/qa762/Desktop/Make_my_profile/yolov5/runs/detect/{self.file_name}/labels/{self.file_name}.txt'
        with open(file_path, 'r') as file:
            # 파일에서 텍스트 읽기
            text = file.read().strip()

        box = text.split()
        print("bbox list:", box)



        yolov5_xmin = int((float(box[1])-float(box[3])/ 2) * self.img.shape[1])
        yolov5_ymin = int((float(box[2]) - float(box[4])/ 2) * self.img.shape[0])
        yolov5_width = int(float(box[3]) * self.img.shape[1])
        yolov5_height = int(float(box[4]) * self.img.shape[0])

        face_img = self.img[yolov5_ymin:yolov5_ymin + yolov5_height, yolov5_xmin:yolov5_xmin + yolov5_width]


        cv2.imwrite(f'{self.file_name}_profile.jpg', face_img)



    def face_img_to_json(self, face_fileName):
        with open(face_fileName, "rb") as image_file:
            image_binary = image_file.read()
            encoded_string = base64.b64encode(image_binary)

            image_dict = {
                "test_image.png": encoded_string.decode()
            }

            # image_json = json.dumps(image_dict)
            image_base64_string = list(image_dict.values())[0]

            return image_base64_string





