import cv2
import numpy as np
import os

# --- Ayarlar ---
NUM_CALIBRATION_POINTS = 4
CAMERA_WINDOW_NAME = "Kamera Goruntusu - Kalibrasyon Noktalarini Secin"
SENSOR_GRID_WINDOW_NAME = "Sensor Grid (8x8) - Referans Kosesini Sec (SolUst,SagUst,SolAlt,SagAlt)"
CAMERA_RESOLUTION_WIDTH = 480
CAMERA_RESOLUTION_HEIGHT = 480
SENSOR_GRID_DISPLAY_SIZE = 256
CALIBRATION_FILE = "calibration_matrix.npy"

# Global değişkenler
camera_points_collected = []
sensor_logical_points_to_match = []

TARGET_SENSOR_LOGICAL_POINTS_ORDERED = [
    (0, 0),  # Sol Üst
    (7, 0),  # Sağ Üst
    (0, 7),  # Sol Alt
    (7, 7)   # Sağ Alt
]
TARGET_SENSOR_NAMES = ["Sol Ust", "Sag Ust", "Sol Alt", "Sag Alt"]

def select_camera_point_callback(event, x, y, flags, param):
    global camera_points_collected, sensor_logical_points_to_match, frame_display_global

    if event == cv2.EVENT_LBUTTONDOWN:
        current_selection_index = len(camera_points_collected)
        if current_selection_index < NUM_CALIBRATION_POINTS:
            camera_points_collected.append((x, y))
            sensor_logical_points_to_match.append(TARGET_SENSOR_LOGICAL_POINTS_ORDERED[current_selection_index])
            
            print(f"Kamera noktasi ({x},{y}) <-> Mantiksal Sensor noktasi {TARGET_SENSOR_LOGICAL_POINTS_ORDERED[current_selection_index]} ({TARGET_SENSOR_NAMES[current_selection_index]}) eklendi.")
            print(f"Kalan nokta: {NUM_CALIBRATION_POINTS - len(camera_points_collected)}")
            
            cv2.circle(frame_display_global, (x,y), 5, (0,255,0), -1)
            cv2.putText(frame_display_global, str(len(camera_points_collected)), (x+7, y+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow(CAMERA_WINDOW_NAME, frame_display_global)
        else:
            print("Maksimum kamera noktasi sayisina ulasildi.")

def draw_sensor_grid_reference_display():
    grid_img = np.ones((SENSOR_GRID_DISPLAY_SIZE, SENSOR_GRID_DISPLAY_SIZE, 3), dtype=np.uint8) * 255
    cell_size = SENSOR_GRID_DISPLAY_SIZE // 8
    for i in range(9):
        cv2.line(grid_img, (i * cell_size, 0), (i * cell_size, SENSOR_GRID_DISPLAY_SIZE -1), (150,150,150), 1)
        cv2.line(grid_img, (0, i * cell_size), (SENSOR_GRID_DISPLAY_SIZE -1, i * cell_size), (150,150,150), 1)
    
    cv2.circle(grid_img, (0*cell_size, 0*cell_size), 5, (0,0,255), -1) # SU
    cv2.putText(grid_img, "SU(0,0)", (0*cell_size + 7, 0*cell_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
    cv2.circle(grid_img, (8*cell_size -1, 0*cell_size), 5, (0,255,0), -1) # SĞU
    cv2.putText(grid_img, "SgU(7,0)", (7*cell_size - 20, 0*cell_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
    cv2.circle(grid_img, (0*cell_size, 8*cell_size-1), 5, (255,0,0), -1) # SA
    cv2.putText(grid_img, "SA(0,7)", (0*cell_size + 7, 7*cell_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0),1)
    cv2.circle(grid_img, (8*cell_size-1, 8*cell_size-1), 5, (0,255,255), -1) #SĞA
    cv2.putText(grid_img, "SgA(7,7)", (7*cell_size - 20, 7*cell_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
    return grid_img

print("--- Kalibrasyon Scripti ---")
print("Lutfen kameranin ve sensorun son pozisyonlarinda SABITLENMIS oldugundan emin olun.")
print("ABCD yuzeyine, sensorun 8x8 grid'inin koselerine denk gelecek sekilde 4 adet belirgin HEDEF ISARETI koyun.")
print("Bu isaretler, kameranin net gorebilecegi sekilde olmalidir.")
print("---------------------------------")
print("Adimlar:")
print("1. 'Sensor Grid' penceresi, hangi mantiksal sensor kosesini hedeflediginizi gosterir.")
print("2. 'Kamera Goruntusu' penceresinde, gosterilen bu mantiksal sensor kosesinin,")
print("   ABCD yuzeyine koydugunuz HEDEF ISARETLERINDEN hangisine denk geldigini FARE ILE TIKLAYIN.")
print(f"   Toplam {NUM_CALIBRATION_POINTS} nokta cifti (SolUst, SagUst, SolAlt, SagAlt sirasiyla) secilecek.")
print("   Hata yaparsaniz 'r' tusu ile resetleyebilirsiniz.")
print("   Tamamlandiginda matris kaydedilecek ve dogrulama gridi cizilecektir. 'q' ile cikin.")
print("---------------------------------")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("HATA: Kamera acilamadi!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow(CAMERA_WINDOW_NAME)
cv2.setMouseCallback(CAMERA_WINDOW_NAME, select_camera_point_callback)

sensor_grid_ref_image = draw_sensor_grid_reference_display()
cv2.imshow(SENSOR_GRID_WINDOW_NAME, sensor_grid_ref_image)

frame_display_global = np.zeros((CAMERA_RESOLUTION_HEIGHT, CAMERA_RESOLUTION_WIDTH, 3), dtype=np.uint8)
perspective_matrix = None
calibration_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hata: Kameradan frame okunamadi.")
        time.sleep(0.1)
        continue

    frame_display_global = frame.copy()
    for i, pt in enumerate(camera_points_collected):
        cv2.circle(frame_display_global, pt, 5, (0,255,0), -1)
        cv2.putText(frame_display_global, str(i+1), (pt[0]+7, pt[1]+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    current_selection_index = len(camera_points_collected)
    if not calibration_done and current_selection_index < NUM_CALIBRATION_POINTS:
        target_name = TARGET_SENSOR_NAMES[current_selection_index]
        target_coords = TARGET_SENSOR_LOGICAL_POINTS_ORDERED[current_selection_index]
        status_text = f"Kamerada TIKLA: Sensorun '{target_name}' ({target_coords[0]},{target_coords[1]}) kosesi ({current_selection_index + 1}/{NUM_CALIBRATION_POINTS})"
        cv2.putText(frame_display_global, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    elif calibration_done:
        cv2.putText(frame_display_global, "Kalibrasyon Tamam! Matris Kaydedildi.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_display_global, "Dogrulama Gridi Cizildi. 'q' ile cik, 'r' ile resetle.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Tüm noktalar seçildiyse ve matris henüz hesaplanmadıysa hesapla
    if len(camera_points_collected) == NUM_CALIBRATION_POINTS and perspective_matrix is None:
        print("\nTum noktalar secildi. Perspektif donusum matrisi hesaplaniyor...")    
        pts_sensor_logical = np.float32(sensor_logical_points_to_match)
        pts_camera_pixels = np.float32(camera_points_collected)

        try:
            perspective_matrix = cv2.getPerspectiveTransform(pts_sensor_logical, pts_camera_pixels)
            print("Perspektif Donusum Matrisi:")
            print(perspective_matrix)
            np.save(CALIBRATION_FILE, perspective_matrix)
            print(f"Matris '{CALIBRATION_FILE}' olarak kaydedildi.")
            calibration_done = True
        except Exception as e:
            print(f"HATA: Matris hesaplanamadi: {e}")
            print("Lutfen noktalari dogru sirada ve duzgun sectiginize emin olun (noktalar ayni dogrultuda olmamali).")
            print("'r' tusuna basarak resetleyin.")
            camera_points_collected = []
            sensor_logical_points_to_match = []
            perspective_matrix = None
            calibration_done = False

    if perspective_matrix is not None:
        s_coords_for_grid = np.zeros((9*9, 2), dtype=np.float32)
        idx = 0
        for r_idx in range(9):
            for c_idx in range(9):
                s_coords_for_grid[idx] = [float(c_idx), float(r_idx)]
                idx += 1
        
        transformed_grid_camera_pixels = cv2.perspectiveTransform(s_coords_for_grid.reshape(-1, 1, 2), perspective_matrix)
        for r_idx in range(9):
            for c_idx in range(9):
                current_pt_on_cam_idx = r_idx * 9 + c_idx
                pt_on_cam = (int(transformed_grid_camera_pixels[current_pt_on_cam_idx][0][0]), \
                             int(transformed_grid_camera_pixels[current_pt_on_cam_idx][0][1]))
                
                if c_idx < 8:
                    next_pt_hor_on_cam_idx = r_idx * 9 + (c_idx + 1)
                    pt_on_cam_next_hor = (int(transformed_grid_camera_pixels[next_pt_hor_on_cam_idx][0][0]), \
                                          int(transformed_grid_camera_pixels[next_pt_hor_on_cam_idx][0][1]))
                    cv2.line(frame_display_global, pt_on_cam, pt_on_cam_next_hor, (255, 100, 100), 1)
                
                if r_idx < 8:
                    next_pt_ver_on_cam_idx = (r_idx + 1) * 9 + c_idx
                    pt_on_cam_next_ver = (int(transformed_grid_camera_pixels[next_pt_ver_on_cam_idx][0][0]), \
                                          int(transformed_grid_camera_pixels[next_pt_ver_on_cam_idx][0][1]))
                    cv2.line(frame_display_global, pt_on_cam, pt_on_cam_next_ver, (255, 100, 100), 1)

    cv2.imshow(CAMERA_WINDOW_NAME, frame_display_global)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        camera_points_collected = []
        sensor_logical_points_to_match = []
        perspective_matrix = None
        calibration_done = False
        print("Kalibrasyon resetlendi. Noktalari yeniden secin.")

cap.release()
cv2.destroyAllWindows()
print("Kalibrasyon scripti sonlandirildi.")
