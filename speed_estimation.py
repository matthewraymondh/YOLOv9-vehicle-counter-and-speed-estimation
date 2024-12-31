import cv2
import torch
import numpy as np
import time
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Define command line flags
flags.DEFINE_string('weights', 'yolov9-c-converted.pt', 'Path to YOLOv9 Weights')
flags.DEFINE_string('video', './highway.mp4', 'Path to input video or webcam index (0)')
flags.DEFINE_string('classes', './coco.names', 'Class Names')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')

def show_fps(frame, fps):    
    x, y, w, h = 10, 10, 330, 45

    # Draw black background rectangle
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), -1)

    # Text FPS
    cv2.putText(frame, "FPS: " + str(fps), (20,52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0,255,0), 3)

def show_counter(frame, title, class_names, vehicle_count, x_init):
    overlay = frame.copy()

    # Show Counters
    y_init = 100
    gap = 30

    alpha = 0.5

    cv2.rectangle(overlay, (x_init - 5, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for vehicle_id, count in vehicle_count.items():
        y_init += gap

        vehicle_name = class_names[vehicle_id]
        vehicle_count = "%.3i" % (count)
        cv2.putText(frame, vehicle_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)            
        cv2.putText(frame, vehicle_count, (x_init + 135, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def show_region(frame, points):
    for id, point in enumerate(points):        
        start_point = (int(points[id-1][0]), int(points[id-1][1]))
        end_point = (int(point[0]), int(point[1]))

        cv2.line(frame, start_point, end_point, (0,0,255), 3)  

def transform_points(perspective, points):
    if points.size == 0:
        return points

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(
            reshaped_points, perspective)
    
    return transformed_points.reshape(-1, 2)     

def add_position_time(track_id, current_position, track_data):
    track_time = time.time()

    if(track_id in track_data):
        track_data[track_id]['position'].append(current_position)
    else:
        track_data[track_id] = {'position' : [current_position], 'time': track_time}

def calculate_speed(start, end, start_time):
    now = time.time()

    move_time = now - start_time    
    distance = abs(end - start)    
    distance = distance / 10

    # m/s
    speed = (distance / move_time)
    # Convert m/s to km/h
    speed = speed * 3.6

    return speed

def speed_estimation(vehicle_position, speed_region, perspective_region, track_data, track_id, text):   
    min_x = int(np.amin(speed_region[:, 0]))
    max_x = int(np.amax(speed_region[:, 0]))

    min_y = int(np.amin(speed_region[:, 1]))
    max_y = int(np.amax(speed_region[:, 1]))

    if((vehicle_position[0] in range(min_x, max_x)) and (vehicle_position[1] in range(min_y, max_y))):
        points = np.array([[vehicle_position[0], vehicle_position[1]]], 
                        dtype=np.float32)                                

        point_transform = transform_points(perspective_region, points)                
        
        add_position_time(track_id, int(point_transform[0][1]), track_data)                

        if(len(track_data[track_id]['position']) > 5):
            start_position = track_data[track_id]['position'][0]
            end_position = track_data[track_id]['position'][-1]
            start_estimate = track_data[track_id]['time']

            speed = calculate_speed(start_position, end_position, start_estimate)
            speed_string = "{:.2f}".format(speed)

            text = text + " - " + speed_string + " km/h"
    
    return text

def main(_argv):
    # Initialize the video capture
    video_input = FLAGS.video
    # Check if the video input is an integer (webcam index)
    if FLAGS.video.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return   

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=5)
    # Select device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    yolov9_weights = FLAGS.weights
    model = DetectMultiBackend(weights=yolov9_weights, device=device, fuse=True)
    model = AutoShape(model)

    # Load the COCO class labels
    classes_path = FLAGS.classes
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

    ## Vehicle Counter
    # Helper Variable
    entered_vehicle_ids = []
    exited_vehicle_ids = []    

    vehicle_class_ids = [1, 2, 3, 5, 7]

    vehicle_entry_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    vehicle_exit_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    
    entry_line = {
        'x1' : 160, 
        'y1' : 558,  
        'x2' : 708,  
        'y2' : 558,          
    }
    exit_line = {
        'x1' : 1155, 
        'y1' : 558,  
        'x2' : 1718,  
        'y2' : 558,          
    }
    offset = 20
    ##

    ## Speed Estimation
    # Region 1 (Left)
    speed_region_1 = np.float32([[393, 478], 
					[760, 482],
					[611, 838], 
					[-135, 777]]) 
    width = 150
    height = 270
    target_1 = np.float32([[0, 0], 
					[width, 0],
					[width, height], 
					[0, height]])
    
    # Region 2 (Right)
    speed_region_2 = np.float32([[1074, 500], 
					[1422, 490],
					[2021, 812], 
					[1377, 932]])     
    width = 120
    height = 270
    target_2 = np.float32([[0, 0], 
					[width, 0],
					[width, height], 
					[0, height]])
    
    # Transform Perspective
    perspective_region_1 = cv2.getPerspectiveTransform(speed_region_1, target_1)    
    perspective_region_2 = cv2.getPerspectiveTransform(speed_region_2, target_2)

    track_data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Run model on each frame
        results = model(frame)

        # Counting Line
        cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)

        # Speed Region
        show_region(frame, speed_region_1)
        show_region(frame, speed_region_2)

        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]            
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Filter out weak detections by confidence threshold
            if confidence < FLAGS.conf:
                continue        

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue            
            
            track_id = track.track_id
            ltrb = track.to_ltrb()            
            x1, y1, x2, y2 = map(int, ltrb)    
            class_id = track.get_det_class()
            color = colors[class_id]
            B, G, R = map(int, color)

            text = f"{track_id} - {class_names[class_id]}"            
            
            center_x = int((x1 + x2) / 2 )
            center_y = int((y1 + y2) / 2 )

            ## Speed Estimation
            # Region 1  
            vehicle_position = (center_x, y2)
            text = speed_estimation(vehicle_position, speed_region_1, perspective_region_1, track_data, track_id, text)   

            # Region 2  
            vehicle_position = (center_x, y1)
            text = speed_estimation(vehicle_position, speed_region_2, perspective_region_2, track_data, track_id, text)            
            ##

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Counter in
            if((center_x in range(entry_line['x1'], entry_line['x2'])) and (center_y in range(entry_line['y1'], entry_line['y1'] + offset)) ):            
                if(int(track_id) not in entered_vehicle_ids and class_id in vehicle_class_ids):                    
                    vehicle_entry_count[class_id] += 1
                    entered_vehicle_ids.append(int(track_id))                

            # Counter out
            if((center_x in range(exit_line['x1'], exit_line['x2'])) and (center_y in range(exit_line['y1'] - offset, exit_line['y1'])) ):                        
                if(int(track_id) not in exited_vehicle_ids and class_id in vehicle_class_ids):                    
                    vehicle_exit_count[class_id] += 1                  
                    exited_vehicle_ids.append(int(track_id)) 
        
        # Show Counters
        show_counter(frame, "Vehicle Enter", class_names, vehicle_entry_count, 10)
        show_counter(frame, "Vehicle Exit", class_names, vehicle_exit_count, 1710)

        end_time = time.time()

        # FPS Calculation
        fps = 1 / (end_time - start_time)
        fps = float("{:.2f}".format(fps))        
        # Show FPS
        show_fps(frame, fps)

        resized = cv2.resize(frame, (1280, 720))
        cv2.imshow('YOLOv9 Object tracking', resized)        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        

    # Release video capture
    cap.release()    

if __name__ == '__main__':
    app.run(main)
