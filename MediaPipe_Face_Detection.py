import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

faceDetection = mp_face_detection.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.5
)

while True:
    ret, img = cap.read()
    
    if ret:
        # BGR => RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = faceDetection.process(imgRGB) 


    if result.detections:
        # detect more than 5 people and bbox turn to red.
        if len(result.detections) > 5:
            for detection in result.detections:
                mp_drawing.draw_detection(
                img, 
                detection,
                bbox_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=7)
                )
                cv2.putText(img, "more than 5", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
 
                # Get the face bounding box and face key points coordinates.
                # Right Eye -1
                # Left Eye -2
                # Nose Tip -3
                # Mouth Center -4
                # Right Ear Tragion -5
                # Left Ear Tragion -6
                for face_no, face in enumerate(result.detections):
                
                    # Display the face number upon which we are iterating upon.
                    # print(f'FACE NUMBER: {face_no+1}')
                    
                    # Display the face confidence.
                    # print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
                    
                    # Get the face bounding box and face key points coordinates.
                    face_data = face.location_data
                    
                    # Display the face bounding box coordinates.
                    # xmin and width => img width   ymin and height => img height
                    # print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')
                    
                    # Iterate key points of each detected face.
                    # for i in range(6):
                        # Display the found normalized key points.
                        # print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
                        # print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}') 
        else:
            for detection in result.detections:
                mp_drawing.draw_detection(
                img, 
                detection,
                bbox_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=7)
                )
                cv2.putText(img, "pass", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)


    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break