from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm
import json

if __name__ == "__main__":
    root_dir = '/home/zhenjue2/wx_ws/data/2501_catl/chajie/'
    labelme_dir = os.path.join(root_dir, 'json_labels')
    image_dir = os.path.join(root_dir, 'images')
    output_dir = os.path.join(root_dir, 'output')
    TN_dir = os.path.join(output_dir, 'TN')
    FP_dir = os.path.join(output_dir, 'FP')
    FN_dir = os.path.join(output_dir, 'FN')
    TP_dir = os.path.join(output_dir, 'TP')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TN_dir, exist_ok=True)
    os.makedirs(FP_dir, exist_ok=True)
    os.makedirs(FN_dir, exist_ok=True)
    os.makedirs(TP_dir, exist_ok=True)

    # root_dir = '/home/zhenjue2/wx_ws/data/2501_catl/chajie/'
    # labelme_dir = os.path.join(root_dir, 'json_labels')
    # image_dir = '/home/zhenjue2/wx_ws/data/2501_catl/chajie/detect_dataset_yolo/images/val'
    
    

    # Load a model
    model = YOLO(
        "/home/zhenjue2/wx_ws/code/2501_catl/ultralytics/self/runs/detect/train6/weights/best.pt")

    cls_model = YOLO(
        '/home/zhenjue2/wx_ws/code/2501_catl/ultralytics/self/runs/classify/train/weights/best.pt')

    all_image_files = os.listdir(image_dir)
    image_files = [file for file in all_image_files if file.endswith('.jpg')]
    num_total = len(image_files)
    num_TP, num_FP, num_TN, num_FN = 0, 0, 0, 0
    for index, image_file in tqdm(enumerate(image_files)):
        json_path = os.path.join(labelme_dir, image_file.replace('.jpg', '.json'))
        image_path = os.path.join(
            image_dir, image_file)
        image = cv2.imread(image_path)
        image_prediction = image.copy()
        image_gt = image.copy()

        # gt
        # 读取 labelme 标注文件
        # 初始都认为是OK品
        ground_truth = True
        prediction = True
        with open(json_path, 'r') as f:
            labelme_data = json.load(f)

        shapes = labelme_data['shapes']
        for index, shape in enumerate(shapes):
            label = shape['label']
            if label == 'button_anomaly':
                ground_truth = False
            # vis gt
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]

            if label == 'button_normal':
                # OK - green
                color = (0, 255, 0)
                text = 'OK'
            else:
                # NG - red
                color = (0, 0, 255)
                text = 'NG'
            cv2.rectangle(image_gt, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image_gt, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # detection
        ret = []
        result = model.predict(image)
        if result is None:
            continue
        result = result[0]
        dets = result.boxes
        for xyxy, conf, clss in zip(dets.xyxy.cpu().numpy().tolist(), dets.conf.cpu().numpy().tolist(), dets.cls.cpu().numpy().tolist()):
            c = int(clss)  # integer class
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

            crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2
            defects = {'type': c,
                       'area': [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                       'score': conf}
            ret.append(defects)

        # classification
        for defect in ret:
            x1, y1, x2, y2 = defect['area']
            region = image[y1:y2, x1:x2]
            result = cls_model.predict(region)
            top1_score, top1_label = result[0].probs.top1conf.cpu().numpy().tolist(), result[0].probs.top1
            # {0: 'anomaly', 1: 'normal'}
            if top1_label == 0:
                prediction = False
        
            # vis prediction
            if top1_label:
                # OK - green
                color = (0, 255, 0)
                text = 'OK'
            else:
                # NG - red
                color = (0, 0, 255)
                text = 'NG'
            cv2.rectangle(image_prediction, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image_prediction, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        cv2.putText(image_gt, 'Ground Truth', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
        cv2.putText(image_prediction, 'Prediction', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
        combined_image = cv2.hconcat([image_gt, image_prediction])
        
        # cv2.imwrite(os.path.join(output_dir, image_file), combined_image)


        if ground_truth and prediction:
            num_TP += 1
            cv2.imwrite(os.path.join(TP_dir, image_file), combined_image)
        elif ground_truth and not prediction:
            num_FN += 1
            cv2.imwrite(os.path.join(FN_dir, image_file), combined_image)
        elif not ground_truth and prediction:
            num_FP += 1
            cv2.imwrite(os.path.join(FP_dir, image_file), combined_image)
        elif not ground_truth and not prediction:
            num_TN += 1
            cv2.imwrite(os.path.join(TN_dir, image_file), combined_image)
        

    print("total:", num_total, "\nTP:", num_TP, "\nFN:", num_FN, "\nTN:", num_TN, "\nFP:", num_FP)
    
    total_accuracy = (num_TP + num_TN)/(num_total)
    positive_recall = num_TP/(num_TP+num_FN)
    positive_precision = num_TP/(num_TP+num_FP)
    negative_recall = num_TN/(num_TN+num_FP)
    negative_precision = num_TN/(num_TN+num_FN)

    print("total_accuracy:", f"{total_accuracy:.3f}", "\npositive_recall:", f"{positive_recall:.3f}", "\npositive_precision:", f"{positive_precision:.3f}", "\nnegative_recall:", f"{negative_recall:.3f}", "\nnegative_precision:", f"{negative_precision:.3f}")
