from ultralytics import YOLO

# Load your YOLOv8 PyTorch model (.pt)
model = YOLO('yolov8s-seg.pt')  # or your specific model

# Export to ONNX with dynamic batch dimension
model.export(format='onnx', dynamic=True, simplify=True, opset=12, imgsz=640)
