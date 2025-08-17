from ultralytics import YOLO

def load_and_train():
    # builds a new model from YOLO default YAML and save the weights/pre-trained model in root directory
    model = (YOLO("yolo11n.yaml").load("yolo11n.pt"))

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=50,
        save_period=5,
    )
    
    model.save("models/yolo11n_custom.pt")


def main():
    print("started")
    load_and_train()


main()