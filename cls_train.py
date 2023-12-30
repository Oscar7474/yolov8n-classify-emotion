from ultralytics import YOLO

# train
model = YOLO("yolov8x-cls.yaml")

if __name__ == "__main__":
    results = model.train(data="C:\Users\Kuo\Downloads\CLS\CLS\Fer2013")
