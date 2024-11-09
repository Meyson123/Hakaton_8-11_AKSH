from ultralytics import YOLO
import asyncio, cv2, os

model = YOLO("2best.pt")


async def convert(file):
    global k
    cap = cv2.VideoCapture(f'videos/{file}')
    if not cap.isOpened():
        print("Ошибка открытия видеофайла")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frameNumber = 0
    frameForSave = 18 * 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Обрабатываем каждый кадр (уменьшаем разрешение)
        frame = cv2.resize(frame, (width, height))
        #Сохраняем фото по условию
        if frameNumber % frameForSave == 0:
            output_file = os.path.join(os.path.abspath('images'),
                                       f'frame_{(frameNumber // frameForSave) + 1:04d}.png')  # Формат: frame_0000.png
            # Сохраняем кадр как отдельное изображение
            cv2.imwrite(output_file, frame)

        frameNumber += 1
    cap.release()


async def main():
    for i in os.listdir(os.path.abspath('videos')):
        await convert(i)
    for i in os.listdir(os.path.abspath('images')):
        model.predict(source=f'images/{i}', show=True, save=True, conf=0.4)


asyncio.run(main())
