import cv2
import numpy as np
import time
import queue
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from sahi import AutoDetectionModel

from config import (
    DEVICE, REGIONS, NUM_BLOCKS, MAX_QUEUE_SIZE, NUM_WORKERS, 
    VEHICLE_CLASSES, RTSP_URLS, PROCESSED_FOLDER, IMAGE_SAVE_INTERVAL, 
    FRAME_SKIP, VEHICLE_THRESHOLD, PROCESS_DURATION
)
import database

# Load models
def load_models():
    try:
        local_model_path = 'models/yolov8m.pt'
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=local_model_path,
            confidence_threshold=0.3,
            device=DEVICE
        )
        yolo_model = YOLO(local_model_path)
        yolo_model.to(DEVICE)
        return detection_model, yolo_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

detection_model, yolo_model = load_models()

class CameraProcessor:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self.region_blocks = self.initialize_region_blocks()
        self.last_save_time = time.time()

    def initialize_region_blocks(self):
        region_data = REGIONS[f'cam{self.cam_id}']['R1']
        return self.divide_region_into_blocks(region_data['vertices'], NUM_BLOCKS)

    @staticmethod
    def is_point_in_polygon(x, y, vertices):
        return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

    @staticmethod
    def divide_region_into_blocks(vertices, num_blocks):
        min_y = min(vertices[:, 1])
        max_y = max(vertices[:, 1])
        block_height = (max_y - min_y) / num_blocks
        return [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]

    def process_frame(self, frame):
        try:
            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=True,
                postprocess_type="NMM",
                postprocess_match_threshold=0.5,
                verbose=0
            )
            return result
        except Exception as e:
            print(f"Error processing frame for camera {self.cam_id}: {e}")
            return None

    def analyze_region_detections(self, detections):
        try:
            region_data = REGIONS[f'cam{self.cam_id}']['R1']
            vertices = region_data['vertices']
            blocks = self.initialize_region_blocks()
            filled_blocks = set()
            vehicles_in_region = 0

            for pred in detections.object_prediction_list:
                if pred.category.name.lower() not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                x_center = int((x1 + x2) / 2.0)
                y_center = int((y1 + y2) / 2.0)

                if self.is_point_in_polygon(x_center, y_center, vertices):
                    vehicles_in_region += 1
                    for idx, (y_min, y_max) in enumerate(blocks):
                        if y_min <= y_center <= y_max:
                            filled_blocks.add(idx)
                            break

            density = (len(filled_blocks) / NUM_BLOCKS) * 100
            return vehicles_in_region, density, filled_blocks
        except Exception as e:
            print(f"Error analyzing region for camera {self.cam_id}: {e}")
            return 0, 0, set()

    def draw_region_visualization(self, frame, vehicles, density, filled_blocks):
        try:
            region_data = REGIONS[f'cam{self.cam_id}']['R1']
            vertices = region_data['vertices']
            color = region_data['color']
            blocks = self.initialize_region_blocks()

            cv2.polylines(frame, [vertices], isClosed=True, color=color, thickness=2)

            for idx, (y_min, y_max) in enumerate(blocks):
                block_color = color if idx in filled_blocks else (128, 128, 128)
                pts = np.array([
                    [vertices[0][0], int(y_min)],
                    [vertices[1][0], int(y_min)],
                    [vertices[1][0], int(y_max)],
                    [vertices[0][0], int(y_max)]
                ], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=block_color, thickness=1)

            text_position = (vertices[0][0], vertices[0][1] - 10)
            cv2.putText(
                frame,
                f'Vehicles: {vehicles}, Density: {density:.1f}%',
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            zebra_data = REGIONS[f'cam{self.cam_id}'].get('Zebra')
            if zebra_data:
                zebra_vertices = zebra_data['vertices']
                zebra_color = zebra_data['color']
                cv2.line(frame, tuple(zebra_vertices[0]), tuple(zebra_vertices[1]), zebra_color, 2)

        except Exception as e:
            print(f"Error visualizing region for camera {self.cam_id}: {e}")

    def process_zebra_crossing_vehicles(self, frame, detections):
        try:
            zebra_data = REGIONS.get(f'cam{self.cam_id}', {}).get('Zebra')
            if not zebra_data:
                return

            vertices = zebra_data['vertices']

            for pred in detections.object_prediction_list:
                if pred.category.name.lower() not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                x_center = int((x1 + x2) / 2.0)
                y_center = int((y1 + y2) / 2.0)

                if self.is_point_in_polygon(x_center, y_center, vertices):
                    database.save_zebra_crossing_data(
                        self.cam_id,
                        vehicle_type=pred.category.name.lower(),
                        x_position=x_center,
                        y_position=y_center,
                        road_side='main',
                        confidence=pred.score.value
                    )

        except Exception as e:
            print(f"Error processing zebra crossing vehicles for camera {self.cam_id}: {e}")

    def analyze_and_save(self, frame, result):
        frame_start_time = time.time()
        processed_frame = frame.copy()
        current_time = datetime.now()

        if result is not None:
            try:
                for pred in result.object_prediction_list:
                    if pred.category.name.lower() not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    score = pred.score.value
                    category = pred.category.name

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f'{category}: {score:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 10
                    cv2.putText(processed_frame, label, (x1, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                r1_vehicles, r1_density, r1_blocks = self.analyze_region_detections(result)
                self.draw_region_visualization(processed_frame, r1_vehicles, r1_density, r1_blocks)

                weighted_vehicles = r1_vehicles
                weighted_density = r1_density

                cv2.putText(processed_frame,
                          f'Total Vehicles: {weighted_vehicles}',
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (255, 255, 255),
                          2)

                # Updated VDC logic: use the dynamic VEHICLE_THRESHOLD variable
                vdc = 0 if r1_vehicles < VEHICLE_THRESHOLD else 1

                processing_time = time.time() - frame_start_time
                database.save_traffic_data(
                    self.cam_id, 
                    r1_vehicles, 
                    r1_density,
                    weighted_vehicles, 
                    weighted_density,
                    vdc, 
                    processing_time
                )

                if time.time() - self.last_save_time >= IMAGE_SAVE_INTERVAL:
                    filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    save_path = os.path.join(PROCESSED_FOLDER, f'cam{self.cam_id}', filename)
                    cv2.imwrite(save_path, processed_frame)
                    self.last_save_time = time.time()

                self.process_zebra_crossing_vehicles(processed_frame, result)
                return processed_frame, weighted_vehicles, weighted_density, vdc

            except Exception as e:
                print(f"Error in analyze_and_save for camera {self.cam_id}: {e}")

        return frame, 0, 0, False

def process_camera_stream(cam_id, duration=PROCESS_DURATION):
    processor = CameraProcessor(cam_id)
    cap = cv2.VideoCapture(RTSP_URLS[f'cam{cam_id}'])

    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        result = processor.process_frame(frame)
        processed_frame, vehicles, density, vdc = processor.analyze_and_save(frame, result)

    cap.release()
    processor.thread_pool.shutdown()

def get_single_frame(cam_id):
    """Get a single frame from a camera for preview"""
    try:
        cap = cv2.VideoCapture(RTSP_URLS.get(cam_id))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        return frame
    except Exception as e:
        print(f"Error capturing single frame from camera {cam_id}: {e}")
        return None

def cyclic_processing():
    """Process each camera in a cycle"""
    while True:
        for cam_id in range(1, 5):
            process_camera_stream(cam_id, PROCESS_DURATION)
            time.sleep(0.1)