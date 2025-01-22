from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/all_data/detect/detect.yaml", epochs=60, imgsz=640, device=[0])
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/old_angle/old_rectangle_7223/detect/old_rectangle_7223.yaml", epochs=55, imgsz=1280, device=[0,1], batch=24)


results = model.train(data="/home/zhenjue2/wx_ws/data/2501_catl/chajie/detect_dataset_yolo/detect.yaml", epochs=500, imgsz=1280, device=[0,1], batch=48)
