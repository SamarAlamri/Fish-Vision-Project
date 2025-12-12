import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)

RESULTS_FOLDER = 'static/results'
SAMPLE_CLASS_FOLDER = 'static/samples/classification'
SAMPLE_DET_FOLDER = 'static/samples/detection'
SAMPLE_SEG_FOLDER = 'static/samples/segmentation'

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_SEG_FOLDER, exist_ok=True) 

CLASSES = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 
    'Red Mullet', 'Red Sea Bream', 'Sea Bass', 
    'Shrimp', 'Striped Red Mullet', 'Trout'
]

# Load Models (Assumes all files exist)
model_cnn = load_model('models/best_model_cnn.h5')
model_mobile = load_model('models/MobileNetV2_fold01_model.h5')
fusion_model = joblib.load('models/fusion_manager_model.pkl')
ann_model = load_model('models/best_ann_model.h5')
model_yolov5 = YOLO('models/yolov5.pt')
model_yolov11 = YOLO('models/yolo11.pt')
model_seg_yolo = YOLO('models/yolov8n-seg.pt') 

cfg = get_cfg()
cfg.merge_from_file("models/config.yaml")
cfg.MODEL.WEIGHTS = "models/mask_cnn_model.pth" 
predictor = DefaultPredictor(cfg)

def get_images_from_folder(folder):
    if not os.path.exists(folder): return []
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_single_hog_features(img_np):
    fd = hog(img_np, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    return fd.reshape(1, -1)

@app.route('/', methods=['GET'])
def index():
    return render_template('interface.html', class_images=get_images_from_folder(SAMPLE_CLASS_FOLDER), det_images=get_images_from_folder(SAMPLE_DET_FOLDER),seg_images=get_images_from_folder(SAMPLE_SEG_FOLDER))

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    selected_task = request.form.get('task')
    selected_image = request.form.get('selected_image_name')

    if not selected_image: return "Error: No image selected.", 400

    if selected_task == 'classification':
        source_folder = SAMPLE_CLASS_FOLDER
    elif selected_task == 'detection':
        source_folder = SAMPLE_DET_FOLDER
    else:
        source_folder = SAMPLE_SEG_FOLDER
        
    filepath = os.path.join(source_folder, selected_image)
    original_web_path = f'samples/{selected_task}/{selected_image}'

    # --- Initialize Variables (Added label_svm) ---
    label_fusion, conf_fusion = None, "N/A"
    label_cnn, conf_cnn = "N/A", "N/A"
    label_mobile, conf_mobile = "N/A", "N/A"
    label_ann, conf_ann = "N/A", "N/A"  
    
    res_filename_v5, res_filename_v11 = None, None
    res_seg_1, res_seg_2 = None, None

    if selected_task == 'classification':
        # 1. Prepare Image
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # 2. Run Deep Learning Models
        preds = model_cnn.predict(img_batch)
        label_cnn = CLASSES[np.argmax(preds)]
        conf_cnn = f"{np.max(preds) * 100:.2f}%"

        preds_m = model_mobile.predict(img_batch)
        label_mobile = CLASSES[np.argmax(preds_m)]
        conf_mobile = f"{np.max(preds_m) * 100:.2f}%"

        # 3. Run ANN
        hog_feats = get_single_hog_features(img_normalized)
        
        if ann_model:
            preds_ann_raw = ann_model.predict(hog_feats)
            ann_idx = np.argmax(preds_ann_raw)
            label_ann = CLASSES[ann_idx]
            conf_ann = f"{np.max(preds_ann_raw) * 100:.2f}%"
            
            # Create One-Hot Vector for Fusion (to match previous logic)
            pred_ann_vec = np.zeros(9)
            pred_ann_vec[ann_idx] = 1.0
        else:
            pred_ann_vec = np.zeros(9)

        # 4. Run Fusion
        meta_features = np.concatenate([pred_ann_vec, preds.flatten(), preds_m.flatten()])
        fusion_result_idx = fusion_model.predict([meta_features])[0]
        fusion_conf = np.max(fusion_model.predict_proba([meta_features]))
        
        label_fusion = fusion_result_idx if isinstance(fusion_result_idx, str) else CLASSES[fusion_result_idx]
        conf_fusion = f"{fusion_conf * 100:.2f}%"

    elif selected_task == 'detection':
        results_v5 = model_yolov5(filepath, imgsz=640)
        res_img_v5 = results_v5[0].plot()
        res_filename_v5 = 'res_v5_' + selected_image
        cv2.imwrite(os.path.join(RESULTS_FOLDER, res_filename_v5), res_img_v5)

        results_v11 = model_yolov11(filepath, imgsz=640)
        res_img_v11 = results_v11[0].plot()
        res_filename_v11 = 'res_v11_' + selected_image
        cv2.imwrite(os.path.join(RESULTS_FOLDER, res_filename_v11), res_img_v11)

    elif selected_task == 'segmentation':
        # 1. YOLOv8 Segmentation (Existing code)
        results = model_seg_yolo(filepath)
        result = results[0]
        
        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            h, w = result.orig_shape
            final_mask = np.zeros((h, w), dtype=np.uint8)
            
            for mask in masks_data:
                resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                final_mask = np.maximum(final_mask, resized_mask)
            
            final_mask = (final_mask * 255).astype(np.uint8)
            res_seg_1 = 'seg_yolo_mask_' + selected_image
            cv2.imwrite(os.path.join(RESULTS_FOLDER, res_seg_1), final_mask)

        im = cv2.imread(filepath)
        outputs = predictor(im)
        
        # Get image dimensions
        h, w = im.shape[:2]
        
        # Create a blank Black canvas (0)
        final_bw_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check if we detected anything
        if outputs["instances"].has("pred_masks"):
            # Get the raw masks (True/False data)
            # Shape is (Number_of_Objects, Height, Width)
            raw_masks = outputs["instances"].pred_masks.to("cpu").numpy()
            
            # Combine all detected objects into one mask
            for mask in raw_masks:
                # 'logical_or' combines overlapping masks
                final_bw_mask = np.logical_or(final_bw_mask, mask)
        
        # Convert True/False (1/0) to White/Black (255/0)
        final_bw_mask = (final_bw_mask.astype(np.uint8) * 255)
        
        # Save the result
        res_seg_2 = 'seg_rcnn_' + selected_image
        cv2.imwrite(os.path.join(RESULTS_FOLDER, res_seg_2), final_bw_mask)

    return render_template(
        'interface.html', 
        show_results=True, task=selected_task, original_web_path=original_web_path,
        label_cnn=label_cnn, conf_cnn=conf_cnn,
        label_mobile=label_mobile, conf_mobile=conf_mobile,
        label_ann=label_ann, conf_ann=conf_ann,
        label_fusion=label_fusion, conf_fusion=conf_fusion,
        res_v5=res_filename_v5, res_v11=res_filename_v11,
        res_seg_1=res_seg_1, res_seg_2=res_seg_2
    )
if __name__ == '__main__':
    app.run(debug=True, port=5000)