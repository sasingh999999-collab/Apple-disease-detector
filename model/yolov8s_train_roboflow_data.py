
# ========================= 2. YOLOv8s RETRAINING SCRIPT =========================
import torch
from ultralytics import YOLO
import os

# Assuming Roboflow downloads the dataset to a folder named 'apple_disease-2'
# You need to adjust the data.yaml path based on the actual downloaded folder name.
# Please inspect your file system to confirm the exact folder name. 
# We'll use a placeholder 'apple_disease-2' and assume the data.yaml is inside.

DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
SAFE_BATCH_SIZE = 210 # Starting with a safer batch size for YOLOv8s on an A40

# ----- GPU Info -----
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"✅ Detected {n_gpus} GPU(s)")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"    Memory Cached:    {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")
else:
    print("⚠ No GPU detected. Training will be very slow on CPU.")

# ----- Optional: Check PyTorch version -----
print(f"PyTorch version: {torch.__version__}")

# ----- YOLOv8 Training -----
model = YOLO("yolov8s.pt")  # Use the small model for faster training

model.train(
    data=DATA_YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=SAFE_BATCH_SIZE, 
    workers=3,          
    project="runs/apple",
    name="hpc_training_aug" # Unique name for this augmented data run
)


print(f"✅ Training started for YOLOv8s with augmented data, logs will be saved in runs/apple/hpc_training_aug/")
