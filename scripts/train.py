import os
import subprocess
import yaml
from clearml import Task
from ultralytics import YOLO


def get_git_info():
    commit = os.getenv("GIT_COMMIT") or subprocess.getoutput("git rev-parse --short HEAD")
    branch = os.getenv("GITHUB_REF") or subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
    return commit.strip(), branch.strip()


def train():
    # Get Git info for tracking
    commit, branch = get_git_info()

    # Initialize ClearML Task
    task = Task.init(
        project_name="Chiller Detection",
        task_name=f"YOLOv8 Training ({commit})",
        task_type=Task.TaskTypes.training
    )
    task.add_tag("ci")
    task.add_tag(branch)
    task.set_parameter("git.commit", commit)
    task.set_parameter("git.branch", branch)

    # Load training parameters from YAML
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    task.connect(params)  # Log hyperparameters

    # Log DVC data reference (optional)
    if os.path.exists("data.dvc"):
        with open("data.dvc") as f:
            task.get_logger().report_text(f.read(), title="data.dvc")

    # Train the YOLOv8 model
    model = YOLO(params["model_size"])
    model.train(
        data="configs/data.yaml",
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
        device="cpu",
        project="models",
        name="chiller_yolov8"
    )

    # Upload best weights to ClearML
    best_model_path = os.path.join("models", "chiller_yolov84", "weights", "best.pt")
    if os.path.exists(best_model_path):
        task.upload_artifact("best_model", best_model_path)
    else:
        task.get_logger().report_text(f"Model not found at {best_model_path}", title="Model Missing")


if __name__ == "__main__":
    train()
