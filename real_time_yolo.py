import torch
import cv2
import argparse
import yaml
import numpy as np
import time
import os
import sys

# Add proper error handling for imports
try:
    from models.yolo import YOLOV1
except ImportError as e:
    print(f"Error importing YOLO model: {e}")
    print("Make sure the models directory is in your Python path and contains yolo.py")
    sys.exit(1)

# Check for available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS acceleration')
print(f"Using device: {device}")

def convert_yolo_pred_x1y1x2y2(yolo_pred, S, B, C, use_sigmoid=False):
    """
    Method converts yolo predictions to x1y1x2y2 format
    """
    out = yolo_pred.reshape((S, S, 5 * B + C))
    if use_sigmoid:
        out[..., :5 * B] = torch.nn.functional.sigmoid(out[..., :5 * B])
    out = torch.clamp(out, min=0., max=1.)
    class_score, class_idx = torch.max(out[..., 5 * B:], dim=-1)

    # Create a grid using these shifts
    shifts_x = torch.arange(0, S, dtype=torch.float32, device=out.device) * 1 / float(S)
    shifts_y = torch.arange(0, S, dtype=torch.float32, device=out.device) * 1 / float(S)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

    boxes = []
    confidences = []
    labels = []
    for box_idx in range(B):
        # xc_offset yc_offset w h -> x1 y1 x2 y2
        boxes_x1 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) -
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y1 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) -
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes_x2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) +
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y2 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) +
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes.append(torch.cat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1))
        confidences.append((out[..., 4 + box_idx * 5] * class_score).reshape(-1))
        labels.append(class_idx.reshape(-1))
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(confidences, dim=0)
    labels = torch.cat(labels, dim=0)
    return boxes, scores, labels

def visualize_detections(image, boxes, labels, scores, label_map, conf_threshold=0.5):
    """
    Draw bounding boxes on image
    """
    # Create a copy of the image to avoid modifying the original
    result_image = image.copy()
    
    # Define colors for different classes (20 classes in VOC)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
              (255, 0, 255), (192, 192, 192), (128, 0, 0), (0, 128, 0), (128, 128, 0),
              (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0),
              (0, 64, 0), (64, 64, 0), (0, 0, 64), (64, 0, 64), (0, 64, 64)]
    
    h, w = image.shape[:2]
    
    # Scale boxes from normalized to actual image size
    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0::2] *= w  # x coordinates
    boxes_scaled[:, 1::2] *= h  # y coordinates
    
    for i, box in enumerate(boxes_scaled):
        if scores[i] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            label_id = int(labels[i].item())
            label = label_map[label_id]
            color = colors[label_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Create text with label and confidence
            text = f"{label}: {scores[i]:.2f}"
            
            # Get size of text for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(result_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            
            # Put text on the background rectangle
            cv2.putText(result_image, text, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return result_image

def try_camera_ids():
    """Try different camera IDs and APIs to find a working one"""
    # Try different backends
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    
    for backend in backends:
        for id in range(3):  # Try first 3 camera IDs
            print(f"Trying camera ID {id} with backend {backend}")
            cap = cv2.VideoCapture(id, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully opened camera {id} with backend {backend}")
                    cap.release()
                    return id, backend
                cap.release()
    
    print("No working camera found. Defaulting to camera 0 with default backend")
    return 0, cv2.CAP_ANY

def safe_nms(boxes, scores, labels, nms_threshold):
    """Apply NMS with error handling"""
    keep_indices = []
    try:
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for class_id in torch.unique(labels):
            curr_indices = torch.where(labels == class_id)[0]
            if len(curr_indices) > 0:
                try:
                    curr_keep_indices = torch.ops.torchvision.nms(
                        boxes[curr_indices],
                        scores[curr_indices],
                        nms_threshold
                    )
                    keep_mask[curr_indices[curr_keep_indices]] = True
                except Exception as e:
                    print(f"NMS error for class {class_id}: {e}")
                    # Fallback: keep all boxes for this class
                    keep_mask[curr_indices] = True
        
        keep_indices = torch.where(keep_mask)[0]
    except Exception as e:
        print(f"Error in NMS: {e}")
        # Fallback: return all indices
        keep_indices = torch.arange(len(scores))
    
    return keep_indices

def main(args):
    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Config file {args.config_path} not found")
        return
    
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error parsing config file: {exc}")
            return
    
    # VOC dataset class labels
    idx2label = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
        10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
        15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }
    
    # Load model
    try:
        dataset_config = config['dataset_params']
        model_config = config['model_params']
        train_config = config['train_params']
        
        yolo_model = YOLOV1(im_size=dataset_config['im_size'],
                            num_classes=dataset_config['num_classes'],
                            model_config=model_config)
    except KeyError as e:
        print(f"Missing key in config file: {e}")
        return
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Check if model file exists
    model_path = args.model_path
    if not model_path:
        task_name = train_config.get('task_name', 'default')
        weights_file = train_config.get('weights_file', 'yolo_voc2007.pth')
        model_path = f"{task_name}/{weights_file}"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    # Load model weights
    try:
        yolo_model.load_state_dict(torch.load(model_path, map_location=device))
        yolo_model.to(device)
        yolo_model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    conf_threshold = args.conf_threshold if args.conf_threshold else train_config.get('infer_conf_threshold', 0.25)
    nms_threshold = train_config.get('nms_threshold', 0.45)
    im_size = dataset_config.get('im_size', 448)
    
    # Initialize video source
    if args.video_path:
        if not os.path.exists(args.video_path):
            print(f"Error: Video file {args.video_path} not found")
            return
        cap = cv2.VideoCapture(args.video_path)
    else:
        camera_id, backend = try_camera_ids()
        print(f"Using camera ID: {camera_id} with backend: {backend}")
        cap = cv2.VideoCapture(camera_id, backend)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    # Test a frame grab
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Failed initial frame grab test")
        return
    else:
        print(f"Successfully grabbed initial frame with shape: {test_frame.shape}")

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Set up FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Starting real-time detection (press 'q' to quit)...")
    print(f"Confidence threshold: {conf_threshold}, NMS threshold: {nms_threshold}")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame, trying again...")
            # Try to reinitialize camera if frame grab fails
            cap.release()
            camera_id, backend = try_camera_ids()
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                print("Camera reinitialization failed. Exiting.")
                break
            continue
            
        # Process frame
        try:
            original_h, original_w = frame.shape[:2]
            
            # Resize frame to model input size
            processed_frame = cv2.resize(frame, (im_size, im_size))
            
            # Convert BGR to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Normalize and convert to tensor
            processed_frame = processed_frame / 255.0
            processed_frame = torch.FloatTensor(processed_frame).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                out = yolo_model(processed_frame)
                
            # Convert prediction to boxes, scores, labels
            boxes, scores, labels = convert_yolo_pred_x1y1x2y2(
                out, 
                S=yolo_model.S, 
                B=yolo_model.B, 
                C=yolo_model.C, 
                use_sigmoid=model_config.get('use_sigmoid', False)
            )
            
            # Apply confidence threshold
            keep = torch.where(scores > conf_threshold)[0]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Apply NMS
            keep = safe_nms(boxes, scores, labels, nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:  # update FPS every 10 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Draw detections
            result_frame = visualize_detections(frame, boxes.cpu(), labels.cpu(), scores.cpu(), idx2label, conf_threshold)
            
            # Add FPS counter
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display
            cv2.imshow('YOLO Real-time Detection', result_frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Fallback: display original frame
            cv2.putText(frame, "Error processing frame", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('YOLO Real-time Detection', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO real-time detection')
    parser.add_argument('--config', dest='config_path', default='config/voc.yaml', type=str,
                        help='Path to config file')
    parser.add_argument('--model', dest='model_path', default=None, type=str,
                        help='Path to model weights (default: from config)')
    parser.add_argument('--camera_id', default=0, type=int,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--video_path', default=None, type=str,
                        help='Path to video file (if not using webcam)')
    parser.add_argument('--conf_threshold', default=None, type=float,
                        help='Confidence threshold (default: from config)')
    
    args = parser.parse_args()
    main(args)