from ultralytics import YOLO


def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="src/yolo/custom_data_seg.yaml",
        imgsz=416,
        epochs=500,
        batch=32,
        workers=4,
        project="runs/segment",
        name="yolov8n_custom_esrgan_data",
    )

    model.val(data="src/yolo/custom_data_seg.yaml")


if __name__ == "__main__":
    main()