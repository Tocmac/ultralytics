from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/home/zhenjue2/wxx_workspace/others/0801_yolov5_qianfeng_bushu/qianfeng/UPDATING-qianfeng_weights/0802-DETECT-CLS-OLD-ANGLE/detect_classify/weights/best.pt")  # load a pretrained model (recommended for training)

# dataset_path = '/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/wrong_samples/label/classify/1209_1216_SDGD_111915_SDGDW1245_1104_guandong_0926_w5_special_color2_0816_0819_w4_w5_all_special_color_0720_0807_all_v2_old_angle_haveseal/dataset'
dataset_path ='/home/zhenjue2/wx_ws/data/2501_catl/chajie/cls_dataset_yolo'

# Train the model with 2 GPUs
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/all_data/detect/detect.yaml", epochs=60, imgsz=640, device=[0])
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/old_angle/old_rectangle_7223_fix_cls_plus0723/old_angle_7223_detect_classify_plus", epochs=30, imgsz=224, device=[1])
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/old_angle/old_rectangle_7223_fix_cls_plus0723_balance0726/old_angle_7223_detect_classify_plus", epochs=20, imgsz=224, device=[1])
# results = model.train(data="/home/zhenjue2/wxx_workspace/data/nezha_qianfeng/wrong_samples/label/classify/0816_0819_w4_w5_all_special_color/dataset", epochs=100, imgsz=224, device=[0])

results = model.train(data=dataset_path, epochs=100, imgsz=224, device=[0], batch=128)



# #### 中断后接续训练 ####
# # Load a model
# model = YOLO("/home/zhenjue2/wxx_workspace/ultralytics-main/self/runs/classify250117-1115_250110_Q4Q5_1205/train/weights/last.pt")     # load a partially trained model
# # Resume training
# results = model.train(resume=True)