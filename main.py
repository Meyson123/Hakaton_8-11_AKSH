import gc
import multiprocessing as mp
from ultralytics import YOLO
import asyncio, cv2, os, shutil
import numpy as np
import pandas as pd

model = YOLO("2best.pt")
df = pd.DataFrame(columns=['Время', 'Наименование столбца'])
if os.path.exists('images'):
    shutil.rmtree('images')
os.makedirs('images')
k = 0
center_line_x = 0
center_line_y = 0


async def convert(file):
    cap = cv2.VideoCapture(f'videos/{file}')
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        exit()

    # Для изменения разрешения фото
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fpsCof = int(-1 * cap.get(cv2.CAP_PROP_FPS) // 1 * -1) // 10

    frameNumber = 0
    frameForSave = 10 * fpsCof

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Обрабатываем каждый кадр (уменьшаем разрешение)
        frame = cv2.resize(frame, (width, height))
        #Сохраняем фото по условию
        if frameNumber % frameForSave == 0:
            outputFile = os.path.join(os.path.abspath('images'),
                                      f'frame_{(frameNumber // frameForSave) + 1:04d}.png')  # Формат: frame_0000.png
            # Сохраняем кадр как отдельное изображение
            cv2.imwrite(outputFile, frame)
        del frame
        gc.collect()

        frameNumber += 1
    cap.release()


async def converter():
    for i in os.listdir(os.path.abspath('videos')):
        await convert(i)


async def predictor(df):
    for i in os.listdir(os.path.abspath('images')):
        results = model.predict(source=f'images/{i}', show=True, save=True, conf=0.4)[0]
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        # Подготовка словаря для группировки результатов по классам
        grouped_objects = {}

        # Группировка результатов
        for class_id, box in zip(classes, boxes):
            class_name = classes_names[int(class_id)]
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box)
        for class_name, details in grouped_objects.items():
            print(f"{class_name}:\n")
            for detail in details:
                if k == 0:
                    x_old, y_old = (detail[0] + detail[1]) // 2, (detail[2] + detail[3]) // 2
                else:
                    x_old, y_old = x, y
                x, y = (detail[0] + detail[1]) // 2, (detail[2] + detail[3]) // 2
                print(f'Object: {x},{y}')
                if class_name == 'sline':
                    print(await sline(x, x_old), df)
                elif class_name == '1.8':
                    print(await abobus(x, center_line_x, y, center_line_y), df)
                elif class_name == 'sve' or class_name == 'sves':
                    print(await stop_line(
                        next((i for i, sublist in enumerate(grouped_objects) if sublist[0] == 'stop'), None),
                        next((i for i, sublist in enumerate(grouped_objects) if sublist[0] == 'car'), None)), df)


async def sline(x, x_old, df):
    if x_old > center_line_x:
        place_old = 'right'
    else:
        place_old = 'left'
    if x > center_line_x:
        place = 'right'
    if x_old != x:
        another_new_row = {"Статья 12.15 часть 4 Выезд в нарушение правил дорожного движения на полосу, предназначенную для встречного движения, при объезде препятствия, либо на трамвайные пути встречного направления, за исключением случаев, предусмотренных частью 3 настоящей статьи"}
        df = df.append(another_new_row, ignore_index=True)


async def abobus(x, center_x, y, center_y, df):
    if center_x - x <= 188:
        if center_y - y <= 190:
            another_new_row = {"Статья 12.17  часть 1.1 и 1.2. движение транспортных средств по полосе для маршрутных транспортных средств или остановка на указанной полосе в нарушение Правил дорожного движения "}
            df = df.append(another_new_row, ignore_index=True)
    else:
        return 'all_good'


async def stop_line(stop, car, df):
    if car == 'None' and stop == 'None':
        another_new_row = {
            "Статья 12.12 часть 2 1. невыполнение требования ПДД об остановке перед стоп-линией, обозначенной дорожными знаками или разметкой проезжей части дороги, при запрещающем сигнале светофора или запрещающем жесте регулировщика"}
        df = df.append(another_new_row, ignore_index=True)


async def main(df):
    await asyncio.gather(predictor(df), converter())


asyncio.run(main(df))
