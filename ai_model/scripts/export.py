import os
from ultralytics import YOLO

def export_model():
    print("[*] Initializing Export Process...")
    weights_path = "../weights/birdbase_v1/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"[!] Target model weights not found at {weights_path}. Using base yolov8n.pt for demonstration.")
        # Fallback for demonstration if actual training hasn't run
        weights_path = "yolov8n.pt"

    print(f"[*] Loading model from {weights_path}")
    model = YOLO(weights_path)

    # 1. Export to ONNX (for Backend deployment)
    print("\n[*] Exporting to ONNX format...")
    onnx_path = model.export(format="onnx", imgsz=640, optimize=True)
    print(f"[*] ONNX model saved at: {onnx_path}")

    # 2. Export to TFLite (for Android offline deployment)
    # This might require TensorFlow installed in the environment (pip install tensorflow)
    print("\n[*] Exporting to TensorFlow Lite (TFLite) format...")
    try:
        tflite_path = model.export(format="tflite", imgsz=640)
        print(f"[*] TFLite model saved at: {tflite_path}")
    except Exception as e:
        print(f"[!] Failed to export to TFLite. Ensure tensorflow is installed: {e}")

    # 3. Export to CoreML (for iOS offline deployment)
    # This might require coremltools installed in the environment (pip install coremltools)
    print("\n[*] Exporting to CoreML format...")
    try:
        coreml_path = model.export(format="coreml", imgsz=640, nms=True)
        print(f"[*] CoreML model saved at: {coreml_path}")
    except Exception as e:
        print(f"[!] Failed to export to CoreML. Ensure coremltools is installed: {e}")

    print("\n[*] Export pipeline finished.")

if __name__ == "__main__":
    export_model()
