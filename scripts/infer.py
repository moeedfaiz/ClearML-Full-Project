import os
from ultralytics import YOLO
from PIL import Image
import cv2

def run_inference():
    model = YOLO("models/chiller_yolov84/weights/best.pt")  # Adjust path as needed

    # Set path to image or directory
    image_dir = "inference/image/image_605466.jpg"
    output_dir = "inference/output"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    if os.path.isdir(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    elif os.path.isfile(image_dir):
        image_paths = [image_dir]
    else:
        print(f"Invalid path: {image_dir}")
        return

    for image_path in image_paths:
        print(f"Inferencing on {image_path}...")
        results = model(image_path)  # returns list of Results objects

        for i, result in enumerate(results):
            # Save annotated image using OpenCV
            annotated_img = result.plot()
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, annotated_img)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_inference()
