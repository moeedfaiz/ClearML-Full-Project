import os
from ultralytics import YOLO
from PIL import Image
import io

# ✅ Ensure the model is pulled via DVC
os.system("dvc pull models/chiller_yolov84/weights/best.pt")

# ✅ Load the YOLOv8 model from DVC-tracked weights
model = YOLO("models/chiller_yolov84/weights/best.pt")

# Example inference function
def run_inference(image_path):
    results = model(image_path)
    for result in results:
        print(result)
        # Optionally save annotated image
        result.save(filename="inference_output.jpg")

if __name__ == "__main__":
    # Example usage
    test_image = "inference/image/image_605466.jpg"
    run_inference(test_image)
