from ultralytics import YOLO


def main():
    model = YOLO("runs/segment/yolov8n_custom_esrgan_data/weights/best.pt")

    model.predict(
        source="sample_data/yolo_test_images",
        imgsz=416,
        conf=0.25,
        save=True,
        project="runs/predict-seg",
        name="demo",
    )


if __name__ == "__main__":
    main()