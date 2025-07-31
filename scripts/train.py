from ultralytics import YOLO
import yaml


def train():
    # Load training parameters from YAML
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load YOLOv8 model (e.g., yolov8n.pt)
    model = YOLO(params["model_size"])

    # Train the model
    model.train(
        data="configs/data.yaml",          # Path to your dataset YAML
        epochs=params["epochs"],           # Number of training epochs
        imgsz=params["imgsz"],             # Image size
        batch=params["batch"],             # Batch size
        device="cpu",                      # Force CPU usage
        project="models",                  # Output folder
        name="chiller_yolov8"              # Run name
    )


if __name__ == "__main__":
    train()
