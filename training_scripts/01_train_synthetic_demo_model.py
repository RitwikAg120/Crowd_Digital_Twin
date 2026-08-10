import warnings, time
warnings.filterwarnings("ignore")
from ultralytics import YOLO

t0 = time.time()
model = YOLO("yolov8n.yaml")
model.train(
    data="synth_dataset/data.yaml",
    epochs=30,
    imgsz=256,
    batch=8,
    device="cpu",
    workers=0,
    patience=30,
    verbose=False,
    project="runs_synth",
    name="cdt_synth2",
    exist_ok=True,
    amp=False,
    plots=False,
    val=True,
    lr0=0.005,
)
print("TRAIN_TIME_SEC", time.time() - t0)
