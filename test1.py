import image_api
from tkinter import filedialog

import time
import cv2
import requests
import json
import subprocess
from pathlib import WindowsPath
from blip_infer import BLIP_INFER

# url = "http://127.0.0.1:7860"
#
# c1 = image_api.ImageApi('download.png')
#
# c1.save_face()
# payload = {
#         "image": f"{c1.face_img_to_json(fr'download.png_face.jpg')}"
#             }
#
# response = requests.post(url=f'{url}/sdapi/v1/interrogate', json=payload)
#
# r = response.text
# prompt = json.loads(r)
#
# prompt = list(prompt.values())[0]
# print(prompt)

from pathlib import WindowsPath
import subprocess
#
#
# command = f"wsl python3 /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/detect.py --save-txt --save-conf --name 22 --weights /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/runs/train/Profile_yolov5s_results/weights/best.pt --conf 0.7 --source /mnt/c/Users/qa762/Desktop/make_my_profile/22.jpg"
# subprocess.run(command, shell=True)

#
# box = list()
# file_path = 'C:/Users/qa762\Desktop/py_project_2023-main/yolov5/runs/detect/exp16/labels/22.txt'
# with open(file_path, 'r') as file:
#     # 파일에서 텍스트 읽기
#     text = file.read().strip()
#
# box = text.split()
# print("박스 리스트:", box[1])


#
#
# c1 = image_api.ImageApi()
# c1.show_image()
# c1.save_profile()
#
# file_path = filedialog.askopenfilename()
# print(file_path)
# file_name = file_path.split('/')
# # print(file_name[5].split('.')[0])
# print(file_name)
# linux_file_path = f'mnt/c/{file_name[1]}/{file_name[2]}/{file_name[3]}/{file_name[4]}/{file_name[5]}'
# print(linux_file_path)
# command = f"wsl python3 /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/detect.py --save-txt --save-conf --name {file_name[5].split('.')[0]} --weights /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/runs/train/Profile_yolov5s_results/weights/best.pt --conf 0.5 --source /{linux_file_path}"
# subprocess.run(command, shell=True)
#
# filename = filedialog.askopenfilename()
# print(filename)
# c1 = image_api.ImageApi(filename)
# c1.save_profile()

#
# c2 = CLIP_INFER(None)
#
# print(c2)
#
# file_path = filename
# file_path = file_path.split('/')
# file_name = file_path[5].split('.')[0] # file name of filepath
# linux_file_path = f'mnt/c/{file_path[1]}/{file_path[2]}/{file_path[3]}/{file_path[4]}/{file_path[5]}' # We are using the Ubuntu terminal, so we need to define 'window_file_path' as 'linux_file_path'
#
# command = f"wsl python3 /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/detect.py --save-txt --save-conf --name {file_name} --weights /mnt/c/Users/qa762/Desktop/make_my_profile/yolov5/runs/train/Profile_yolov5s_results/weights/best.pt --conf 0.5 --source /{linux_file_path}"
# subprocess.run(command, shell=True)
#
#
# filename = filedialog.askopenfilename()
#
# c2 = CLIP_INFER(filename)
# print(c2)
#

c1 = BLIP_INFER('siyo_profile.jpg')

clip_prompt = c1
