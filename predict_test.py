import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import sodhelmet  # noqa: F401  (needed for models with custom layers)
    from ultralytics import YOLO

    model = YOLO(root / "runs/s0_baseline_640/seed0/weights/best.pt")
    model.predict(
        source=r"D:/The Projects/helmet-sod-yolov8/helmet-sod-yolov8/Dataset Indonesia Upper/test/images",
        imgsz=640, conf=0.25, save=True, device=0,
        project=str(root / "runs" / "predict"), name="s0",
    )