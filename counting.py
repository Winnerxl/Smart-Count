import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading, time, json, base64
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
from collections import OrderedDict, deque
import os
from scipy.spatial import distance as dist
import traceback

DEBUG_MODE = False
MQTT_BROKER = "localhost"
MQTT_TOPIC = "sensor/depth"

CALIBRATE_BASE_DEPTH_FRAMES = 40
RELIABLE_MIN_DEPTH_MM_FOR_BASE = 100
RELIABLE_MAX_DEPTH_MM_FOR_BASE = 3800

TARGET_APPLE_HEIGHT_MM = 35
MIN_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM = int(TARGET_APPLE_HEIGHT_MM * 0.75)
MAX_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM = int(TARGET_APPLE_HEIGHT_MM * 1.25)
CALIBRATE_APPLE_HEIGHT_FRAMES = 60
MIN_SAMPLES_FOR_DYN_THRESH_UPDATE = CALIBRATE_APPLE_HEIGHT_FRAMES // 3

MAX_APPLE_CLUSTER_HEIGHT_MM = int(TARGET_APPLE_HEIGHT_MM * 3.5)
MIN_SENSOR_CLUSTER_AREA = 1
IOU_THRESHOLD_FOR_MATCHING = 0.20
STACK_FACTOR_THRESHOLD = 1.5

current_depth_matrix = np.zeros((8, 8), dtype=np.int16)
baseline_calibration_values = deque(maxlen=CALIBRATE_BASE_DEPTH_FRAMES * 2)
BASE_DEPTH_MM = None
PERSPECTIVE_MATRIX = None

apple_height_calibration_samples = deque(maxlen=CALIBRATE_APPLE_HEIGHT_FRAMES)
DYNAMIC_APPLE_THRESHOLD_MM = TARGET_APPLE_HEIGHT_MM

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480
CALIBRATION_FILE_PATH = "calibration_matrix.npy"

DEPTH_COLORMAP = cv2.COLORMAP_JET
NORMALIZE_DEPTH_MIN_MM = 50
NORMALIZE_DEPTH_MAX_MM = 1000

depth_matrix_lock = threading.Lock()

if os.path.exists(CALIBRATION_FILE_PATH):
    try:
        PERSPECTIVE_MATRIX = np.load(CALIBRATION_FILE_PATH)
        print(f"✅ Calibration matrix '{CALIBRATION_FILE_PATH}' loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load calibration matrix: {e}")
        PERSPECTIVE_MATRIX = None
else:
    print(f"WARNING: Calibration matrix '{CALIBRATION_FILE_PATH}' not found.")
    PERSPECTIVE_MATRIX = None

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker.")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"MQTT connection error, Code: {rc}")

def on_message(client, userdata, msg):
    global current_depth_matrix, baseline_calibration_values, BASE_DEPTH_MM
    try:
        depth_values_list = json.loads(msg.payload.decode())
        if isinstance(depth_values_list, list) and len(depth_values_list) == 64:
            temp_mat = np.array(depth_values_list, dtype=np.int16).reshape(8, 8)
            temp_mat[(temp_mat < 10) & (temp_mat != 0)] = 0
            temp_mat[temp_mat > RELIABLE_MAX_DEPTH_MM_FOR_BASE] = 0
            with depth_matrix_lock:
                current_depth_matrix[:] = temp_mat
            if BASE_DEPTH_MM is None:
                border_pixels = np.concatenate([
                    temp_mat[0, :][temp_mat[0, :] != 0], 
                    temp_mat[-1, :][temp_mat[-1, :] != 0],
                    temp_mat[1:-1, 0][temp_mat[1:-1, 0] != 0],
                    temp_mat[1:-1, -1][temp_mat[1:-1, -1] != 0]
                ])
                valid_border_for_calib = border_pixels[(border_pixels >= RELIABLE_MIN_DEPTH_MM_FOR_BASE) &
                                                       (border_pixels <= RELIABLE_MAX_DEPTH_MM_FOR_BASE)]
                if valid_border_for_calib.size > 0:
                    baseline_calibration_values.extend(valid_border_for_calib)
                if len(baseline_calibration_values) >= CALIBRATE_BASE_DEPTH_FRAMES:
                    if baseline_calibration_values:
                        q1 = np.percentile(list(baseline_calibration_values), 25)
                        q3 = np.percentile(list(baseline_calibration_values), 75)
                        iqr_filtered_samples = [s for s in baseline_calibration_values if q1 <= s <= q3]
                        if iqr_filtered_samples:
                            BASE_DEPTH_MM = int(np.median(iqr_filtered_samples))
                            print(f"📏 BASE_DEPTH calibrated: {BASE_DEPTH_MM} mm (Samples: {len(iqr_filtered_samples)})")
                        else:
                            BASE_DEPTH_MM = int(np.median(list(baseline_calibration_values)))
                            print(f"📏 BASE_DEPTH calibrated (no IQR): {BASE_DEPTH_MM} mm (Samples: {len(baseline_calibration_values)})")
                        baseline_calibration_values.clear()
                    else:
                        print("WARNING: Not enough valid data for base depth calibration.")
    except json.JSONDecodeError:
        if DEBUG_MODE: print(f"ERROR: MQTT message is not valid JSON: {msg.payload}")
    except Exception as e:
        if DEBUG_MODE: print(f"ERROR: in on_message: {e}, Payload: {msg.payload if 'msg' in locals() else 'N/A'}")

def mqtt_run_forever():
    while True:
        try:
            mqtt_client.connect(MQTT_BROKER, 1883, 60)
            mqtt_client.loop_forever()
        except Exception as e:
            print(f"MQTT connection error: {e}, reconnecting in 3s...")
            time.sleep(3)

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_thread = threading.Thread(target=mqtt_run_forever, daemon=True)
mqtt_thread.start()

def iou(boxA, boxB):
    if not (boxA and boxB and len(boxA) == 4 and len(boxB) == 4): return 0.0
    if not (boxA[0] < boxA[2] and boxA[1] < boxA[3] and boxB[0] < boxB[2] and boxB[1] < boxB[3]):
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea <= 0 or boxBArea <= 0: return 0.0
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 1e-6 else 0.0

MODEL_PATH, LABEL_PATH = "model/mobilenet_ssd_v2_coco_quant_postprocess.tflite", "model/coco_labels.txt"
labels, apple_idx, interpreter, inp_det, out_det, in_h, in_w, model_input_dtype = [None]*8
try:
    labels_temp = []
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) > 1: labels_temp.append(parts[1])
            elif len(parts) == 1 and parts[0]: labels_temp.append(parts[0])
    if not labels_temp: print(f"ERROR: Label file '{LABEL_PATH}' could not be read or is empty."); exit()
    labels = labels_temp
    if "apple" in labels: apple_idx = labels.index("apple")
    else: apple_idx = -1; print(f"ERROR: 'apple' label not found in '{LABEL_PATH}'!")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()
    input_shape = inp_det['shape']
    in_h = input_shape[1]
    in_w = input_shape[2]
    model_input_dtype = inp_det['dtype']
    print(f"✅ TFLite model '{MODEL_PATH}' loaded. Input size: {in_w}x{in_h}, Dtype: {model_input_dtype}")
except FileNotFoundError: print(f"ERROR: Model '{MODEL_PATH}' or label file '{LABEL_PATH}' not found."); traceback.print_exc(); exit()
except Exception as e: print(f"ERROR: Could not load model/labels: {e}"); traceback.print_exc(); exit()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
HTML_TEMPLATE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Apple Counter + Fusion V2.6 (Improved)</title>
<style>
body{font-family:sans-serif;margin:20px;background-color:#f4f4f4;color:#333}
h3,h4{color:#555}
.container{display:flex;flex-wrap:wrap;gap:30px}
.card{background-color:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}
.card img{border:1px solid #ddd;border-radius:4px}
#feed{max-width:100%;height:auto;display:block;width:""" + str(CAMERA_WIDTH) + """px;height:""" + str(CAMERA_HEIGHT) + """px}
#depthmap{width:256px;height:256px;image-rendering:pixelated}
p{margin:10px 0}
span{font-weight:bold;color:#007bff}
</style></head>
<body>
<h3>Apple Counting + 8×8 Depth Sensor Fusion (Improved)</h3>
<div class="container">
  <div class="card">
    <h4>Camera View</h4>
    <img id="feed" src="" alt="Camera loading..."/>
    <p>Camera Count: <span id="camera_cnt">0</span></p>
    <p>FPS: <span id="fps">0.0</span></p>
  </div>
  <div class="card">
    <h4>Depth Sensor Data (8x8)</h4>
    <img id="depthmap" src="" alt="Depth Map Loading..."/><br/>
    <p>Sensor Total Count: <span id="sensor_total_cnt">0</span></p>
    <p>Sensor-only Count: <span id="sensor_only_cnt">0</span></p>
    <p><b>Total Estimated Apple Count: <span id="total_cnt">0</span></b></p>
    <hr>
    <p><i>Base Depth (BASE_DEPTH): <span id="base_val">CALIBRATING...</span> mm</i></p>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.4.1/socket.io.min.js"></script>
<script>
const sock=io();
sock.on('frame_update',d=>{
  if(document.getElementById('feed'))document.getElementById('feed').src='data:image/jpeg;base64,'+d
});
sock.on('depthmap_update',d=>{
  if(document.getElementById('depthmap'))document.getElementById('depthmap').src='data:image/png;base64,'+d
});
sock.on('stats_update',s=>{
  if(document.getElementById('camera_cnt'))document.getElementById('camera_cnt').innerText=s.camera_count;
  if(document.getElementById('sensor_total_cnt'))document.getElementById('sensor_total_cnt').innerText=s.sensor_total_count;
  if(document.getElementById('sensor_only_cnt'))document.getElementById('sensor_only_cnt').innerText=s.sensor_only_count;
  if(document.getElementById('total_cnt'))document.getElementById('total_cnt').innerText=s.total_count;
  if(document.getElementById('fps'))document.getElementById('fps').innerText=s.fps.toFixed(1);
  if(document.getElementById('base_val'))document.getElementById('base_val').innerText=s.base_depth_mm!==null?s.base_depth_mm+' mm':'CALIBRATING...';
});
</script></body></html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

class CentroidTracker:
    def __init__(self, maxDisappeared=8, maxDistance=CAMERA_WIDTH//3):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
    def register(self, centroid, bbox, class_id):
        self.objects[self.nextObjectID] = (centroid, bbox, class_id)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        if objectID in self.objects: del self.objects[objectID]
        if objectID in self.disappeared: del self.disappeared[objectID]
    def update(self, detected_rects, detected_class_ids):
        if not detected_rects:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects.copy()
        inputCentroids = np.array([((r[0] + r[2]) // 2, (r[1] + r[3]) // 2) for r in detected_rects], dtype="int")
        if not self.objects:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], detected_rects[i], detected_class_ids[i])
            return self.objects.copy()
        objectIDs = list(self.objects.keys())
        if not objectIDs :
            for i in range(len(inputCentroids)):
                 self.register(inputCentroids[i], detected_rects[i], detected_class_ids[i])
            return self.objects.copy()
        objectCentroids = np.array([data[0] for data in self.objects.values()])
        D = dist.cdist(objectCentroids, inputCentroids)
        rows_indices_sorted_by_min_dist = D.min(axis=1).argsort()
        cols_indices_for_best_match = D.argmin(axis=1)[rows_indices_sorted_by_min_dist]
        usedRows = set()
        usedCols = set()
        for (row, col) in zip(rows_indices_sorted_by_min_dist, cols_indices_for_best_match):
            if row in usedRows or col in usedCols: continue
            if D[row, col] > self.maxDistance: continue
            objectID = objectIDs[row]
            self.objects[objectID] = (inputCentroids[col], detected_rects[col], detected_class_ids[col])
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        if D.shape[0] >= D.shape[1]:
            for row_idx in unusedRows:
                objectID = objectIDs[row_idx]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
        else:
            for col_idx in unusedCols:
                self.register(inputCentroids[col_idx], detected_rects[col_idx], detected_class_ids[col_idx])
        return self.objects.copy()

def capture_and_process_thread():
    global BASE_DEPTH_MM, DYNAMIC_APPLE_THRESHOLD_MM, apple_height_calibration_samples, PERSPECTIVE_MATRIX
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("ERROR: Could not open camera."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ct = CentroidTracker()
    frame_counter, time_start_fps, fps_val = 0, time.time(), 0.0
    last_model_rects, last_model_cids = [], []
    print("Image processing and fusion loop started.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame. Retrying...")
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("ERROR: Could not re-open camera.")
                break
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        frame_counter += 1
        display_frame = frame.copy()
        if frame_counter % 2 == 0:
            small_frame = cv2.resize(frame, (in_w, in_h))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb_frame, axis=0).astype(model_input_dtype)
            if model_input_dtype != np.uint8: input_data /= 255.0
            interpreter.set_tensor(inp_det['index'], input_data)
            interpreter.invoke()
            boxes_norm = interpreter.get_tensor(out_det[0]['index'])[0]
            cids_model = interpreter.get_tensor(out_det[1]['index'])[0].astype(int)
            scores_model = interpreter.get_tensor(out_det[2]['index'])[0]
            current_model_rects = []
            current_model_cids = []
            for i in range(len(scores_model)):
                if scores_model[i] >= 0.45:
                    ymin, xmin, ymax, xmax = boxes_norm[i]
                    px_x1 = int(xmin * CAMERA_WIDTH)
                    px_y1 = int(ymin * CAMERA_HEIGHT)
                    px_x2 = int(xmax * CAMERA_WIDTH)
                    px_y2 = int(ymax * CAMERA_HEIGHT)
                    
                    if cids_model[i] == apple_idx:
                        box_height = ymax - ymin
                        if box_height > 0:
                            target_width = box_height * 1.1  # Slightly wider than tall
                            current_width = xmax - xmin
                            if current_width < target_width:
                                width_diff = target_width - current_width
                                xmax = min(1.0, xmin + target_width)
                                px_x2 = int(xmax * CAMERA_WIDTH)
                    
                    px_x1 = max(0, min(px_x1, CAMERA_WIDTH - 1))
                    px_y1 = max(0, min(px_y1, CAMERA_HEIGHT - 1))
                    px_x2 = max(0, min(px_x2, CAMERA_WIDTH - 1))
                    px_y2 = max(0, min(px_y2, CAMERA_HEIGHT - 1))
                    if px_x1 < px_x2 and px_y1 < px_y2:
                        current_model_rects.append((px_x1, px_y1, px_x2, px_y2))
                        current_model_cids.append(cids_model[i])
            last_model_rects, last_model_cids = current_model_rects, current_model_cids
        tracked_objects = ct.update(last_model_rects, last_model_cids)
        camera_detected_apple_bboxes = [data[1] for _, data in tracked_objects.items() if data[2] == apple_idx]
        camera_apple_count = len(camera_detected_apple_bboxes)
        for x1_tr, y1_tr, x2_tr, y2_tr in (data[1] for _, data in tracked_objects.items()):
            cv2.rectangle(display_frame, (x1_tr, y1_tr), (x2_tr, y2_tr), (0, 0, 255), 2)

        if BASE_DEPTH_MM is not None and PERSPECTIVE_MATRIX is not None and camera_detected_apple_bboxes:
            current_frame_apple_heights_for_calib = []
            for cam_bbox in camera_detected_apple_bboxes:
                mask_on_sensor = np.zeros((8,8), dtype=bool)
                sensor_pixels_in_bbox_count = 0
                for r_s in range(8):
                    for c_s in range(8):
                        s_cell_corners = np.float32([[c_s,r_s],[c_s+1,r_s],[c_s,r_s+1],[c_s+1,r_s+1]]).reshape(-1,1,2)
                        try:
                            cam_cell_corners = cv2.perspectiveTransform(s_cell_corners, PERSPECTIVE_MATRIX)
                            if cam_cell_corners is None: continue
                            xc_proj, yc_proj = cam_cell_corners[:,0,0], cam_cell_corners[:,0,1]
                            proj_cell_bbox = (int(np.min(xc_proj)), int(np.min(yc_proj)), int(np.max(xc_proj)), int(np.max(yc_proj)))
                            if iou(proj_cell_bbox, cam_bbox) > 0.05:
                                mask_on_sensor[r_s, c_s] = True
                                sensor_pixels_in_bbox_count += 1
                        except Exception: continue
                if sensor_pixels_in_bbox_count > 0:
                    with depth_matrix_lock:
                        depth_values_in_mask = current_depth_matrix[mask_on_sensor]
                        valid_depths = depth_values_in_mask[(depth_values_in_mask > 0) & (depth_values_in_mask < BASE_DEPTH_MM)]
                    if valid_depths.size > 0:
                        median_obj_depth = np.median(valid_depths)
                        apple_h = BASE_DEPTH_MM - median_obj_depth
                        if MIN_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM <= apple_h <= MAX_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM:
                            current_frame_apple_heights_for_calib.append(apple_h)
            if current_frame_apple_heights_for_calib:
                apple_height_calibration_samples.append(np.median(current_frame_apple_heights_for_calib))
                if DEBUG_MODE: print(f"Apple heights for calibration: {current_frame_apple_heights_for_calib}, deque: {list(apple_height_calibration_samples)}")
        if len(apple_height_calibration_samples) >= MIN_SAMPLES_FOR_DYN_THRESH_UPDATE:
            valid_samples_for_update = [s for s in apple_height_calibration_samples
                                        if MIN_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM <= s <= MAX_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM]
            if len(valid_samples_for_update) >= MIN_SAMPLES_FOR_DYN_THRESH_UPDATE // 2:
                new_dynamic_threshold = int(np.median(valid_samples_for_update))
                DYNAMIC_APPLE_THRESHOLD_MM = max(MIN_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM, min(new_dynamic_threshold, MAX_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM))
                if DEBUG_MODE: print(f"DYNAMIC_APPLE_THRESHOLD_MM updated: {DYNAMIC_APPLE_THRESHOLD_MM} mm (Samples: {len(valid_samples_for_update)})")
        sensor_only_calculated_count = 0
        if BASE_DEPTH_MM is not None:
            with depth_matrix_lock:
                depth_diff = BASE_DEPTH_MM - current_depth_matrix
                object_mask_sensor = ((depth_diff >= MIN_APPLE_HEIGHT_FOR_DYNAMIC_THRESHOLD_MM) &
                                      (depth_diff < MAX_APPLE_CLUSTER_HEIGHT_MM) &
                                      (current_depth_matrix > 0)).astype(np.uint8)
                num_labels, labels_map, stats, centroids_sensor = cv2.connectedComponentsWithStats(object_mask_sensor, connectivity=8)
            sensor_only_display_bboxes = []
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < MIN_SENSOR_CLUSTER_AREA: continue
                s_x, s_y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                s_w, s_h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                projected_sensor_bbox = None
                if PERSPECTIVE_MATRIX is not None:
                    s_corners = np.float32([[s_x,s_y],[s_x+s_w,s_y],[s_x,s_y+s_h],[s_x+s_w,s_y+s_h]]).reshape(-1,1,2)
                    try:
                        c_corners = cv2.perspectiveTransform(s_corners, PERSPECTIVE_MATRIX)
                        if c_corners is not None:
                            xc_proj, yc_proj = c_corners[:,0,0], c_corners[:,0,1]
                            cx1,cy1 = max(0,min(int(np.min(xc_proj)),CAMERA_WIDTH-1)), max(0,min(int(np.min(yc_proj)),CAMERA_HEIGHT-1))
                            cx2,cy2 = max(0,min(int(np.max(xc_proj)),CAMERA_WIDTH-1)), max(0,min(int(np.max(yc_proj)),CAMERA_HEIGHT-1))
                            if cx1 < cx2 and cy1 < cy2: projected_sensor_bbox = (cx1,cy1,cx2,cy2)
                    except Exception: projected_sensor_bbox = None
                if projected_sensor_bbox is None:
                    px_w, px_h = CAMERA_WIDTH / 8, CAMERA_HEIGHT / 8
                    projected_sensor_bbox = (int(s_x*px_w), int(s_y*px_h), int((s_x+s_w)*px_w), int((s_y+s_h)*px_h))
                is_seen_by_cam = any(iou(projected_sensor_bbox, cam_bbox) >= IOU_THRESHOLD_FOR_MATCHING
                                     for cam_bbox in camera_detected_apple_bboxes)
                if not is_seen_by_cam:
                    sensor_only_display_bboxes.append(projected_sensor_bbox)
                    cluster_mask_current = (labels_map == i)
                    with depth_matrix_lock:
                        depth_diffs_in_cluster = depth_diff[cluster_mask_current & (current_depth_matrix > 0)]
                    if depth_diffs_in_cluster.size == 0: continue
                    avg_cluster_height = np.mean(depth_diffs_in_cluster)
                    num_apples_in_sensor_cluster = 1
                    if DYNAMIC_APPLE_THRESHOLD_MM > 0:
                        estimated_stack_count = avg_cluster_height / DYNAMIC_APPLE_THRESHOLD_MM
                        num_apples_in_sensor_cluster = max(1, int(round(estimated_stack_count)))
                        if estimated_stack_count > num_apples_in_sensor_cluster + (STACK_FACTOR_THRESHOLD - 1.0) - 0.1:
                            num_apples_in_sensor_cluster +=1
                    sensor_only_calculated_count += num_apples_in_sensor_cluster
                    if DEBUG_MODE:
                        print(f"SENSOR-ONLY: Proj={projected_sensor_bbox}, AvgH={avg_cluster_height:.0f}, DynThr={DYNAMIC_APPLE_THRESHOLD_MM}, EstApples={num_apples_in_sensor_cluster}")
        for sx1, sy1, sx2, sy2 in sensor_only_display_bboxes:
            cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (255, 100, 0), 2)
        sensor_total_count = sensor_only_calculated_count
        sensor_only_count = sensor_total_count - camera_apple_count
        total_apple_count_new = sensor_only_count + camera_apple_count
        ret_jpg, jpg_buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_jpg: socketio.emit('frame_update', base64.b64encode(jpg_buf).decode('ascii'))
        if current_depth_matrix.size == 64:
            with depth_matrix_lock:
                valid_depth_pixels = current_depth_matrix[current_depth_matrix > 0]
                if valid_depth_pixels.size > 0:
                    display_min_depth = NORMALIZE_DEPTH_MIN_MM
                    display_max_depth = NORMALIZE_DEPTH_MAX_MM
                    clipped_depth = np.clip(current_depth_matrix, display_min_depth, display_max_depth)
                    norm_depth_display = np.zeros_like(clipped_depth, dtype=np.float32)
                    mask_valid = clipped_depth > 0
                    if np.any(mask_valid):
                        norm_depth_display[mask_valid] = cv2.normalize(
                            clipped_depth[mask_valid], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        ).ravel()
                    vis_depth = cv2.applyColorMap(norm_depth_display.astype(np.uint8), DEPTH_COLORMAP)
                    vis_depth_resized = cv2.resize(vis_depth,(256,256),interpolation=cv2.INTER_NEAREST)
                else:
                    vis_depth_resized = np.zeros((256,256,3), dtype=np.uint8)
            ret_png, png_buf = cv2.imencode('.png', vis_depth_resized)
            if ret_png: socketio.emit('depthmap_update', base64.b64encode(png_buf).decode('ascii'))
        elapsed_time_fps = time.time() - time_start_fps
        if elapsed_time_fps >= 1.0:
            fps_val = frame_counter / elapsed_time_fps
            frame_counter = 0
            time_start_fps = time.time()
        socketio.emit('stats_update', {
            'camera_count': camera_apple_count,
            'sensor_total_count': sensor_total_count,
            'sensor_only_count': sensor_only_count,
            'total_count': total_apple_count_new,
            'fps': fps_val,
            'base_depth_mm': BASE_DEPTH_MM
        })
        socketio.sleep(0.005)
    cap.release()
    cv2.destroyAllWindows()
    print("Camera loop finished.")

if __name__ == '__main__':
    if interpreter is None or apple_idx == -1:
        print("Model or 'apple' label could not be loaded, exiting.")
        exit()
    if PERSPECTIVE_MATRIX is None:
        print("WARNING: Perspective calibration matrix not loaded. Simple scaling will be used for sensor projections.")
    host_ip = "0.0.0.0"
    print(f"Web server starting: http://{host_ip}:5000")
    processing_thread = threading.Thread(target=capture_and_process_thread, daemon=True)
    processing_thread.start()
    try:
        run_kwargs = {'app': app, 'host': host_ip, 'port': 5000, 'debug': DEBUG_MODE}
        run_kwargs['use_reloader'] = False
        socketio.run(**run_kwargs)
    except KeyboardInterrupt:
        print("Program terminated by user...")
    except Exception as e_main:
        print(f"Main program error: {e_main}")
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if mqtt_client.is_connected():
            mqtt_client.loop_stop(force=False)
            mqtt_client.disconnect()
            print("MQTT disconnected.")
        print("Program terminated.")
