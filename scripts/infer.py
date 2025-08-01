import os
import subprocess
from clearml import Task
from ultralytics import YOLO
import cv2

def get_git_info():
    commit = os.getenv("GIT_COMMIT") or subprocess.getoutput("git rev-parse --short HEAD")
    branch = os.getenv("GITHUB_REF") or subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
    return commit.strip(), branch.strip()

def run_inference(source_path):
    # Git info for tagging
    commit, branch = get_git_info()

    # Initialize ClearML inference task
    task = Task.init(
        project_name="Chiller Detection",
        task_name=f"YOLOv8 Inference ({commit})",
        task_type=Task.TaskTypes.inference
    )
    task.add_tag("ci")
    task.add_tag(branch)
    task.set_parameter("git.commit", commit)
    task.set_parameter("git.branch", branch)

    # Ensure model is available via DVC
    os.system("dvc pull models/chiller_yolov85/weights/best.pt")

    # Load YOLO model
    model = YOLO("models/chiller_yolov85/weights/best.pt")

    # Determine input(s)
    image_paths = []
    if os.path.isdir(source_path):
        image_paths = [
            os.path.join(source_path, f)
            for f in os.listdir(source_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    elif os.path.isfile(source_path):
        image_paths = [source_path]
    else:
        raise FileNotFoundError(f"Input path not found: {source_path}")

    output_dir = "inference_output"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        print(f"Inferencing on {image_path}...")
        results = model(image_path)

        # Log raw predictions
        try:
            task.get_logger().report_text(
                results[0].tojson(), 
                title=f"Raw Predictions - {os.path.basename(image_path)}"
            )
        except Exception:
            pass

        # Save annotated image and report to ClearML
        for result in results:
            annotated = result.plot()
            out_filename = os.path.basename(image_path)
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, annotated)
            print(f"Saved annotated image to {out_path}")
            task.get_logger().report_image(
                title=f"Inference Result - {out_filename}",
                series=out_filename,
                local_path=out_path
            )

if __name__ == "__main__":
    test_input = "inference/image/image_605466.jpg"
    run_inference(test_input)
