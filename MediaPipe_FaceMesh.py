import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
faceMash = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
baseDot_drawing_spec = mpDraw.DrawingSpec(color=(225,111,35),thickness=1, circle_radius=1)
baseLine_drawing_spec = mp_drawing_styles.DrawingSpec(color=(225,111,35),thickness=3, circle_radius=1)


while True:
  ret, img = cap.read()
  if ret:
    # BGR => RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMash.process(imgRGB)
    # print(result.multi_face_landmarks)

    # Draw the face mesh annotations on the image.
        # connections – options:
        # mp_face_mesh.FACEMESH_FACE_OVAL, 
        # mp_face_mesh.FACEMESH_LEFT_EYE, 
        # mp_face_mesh.FACEMESH_LEFT_EYEBROW, 
        # mp_face_mesh.FACEMESH_LIPS, 
        # mp_face_mesh.FACEMESH_RIGHT_EYE, 
        # mp_face_mesh.FACEMESH_RIGHT_EYEBROW, 
        # mp_face_mesh.FACEMESH_TESSELATION, 
        # mp_face_mesh.FACEMESH_CONTOURS.
    if result.multi_face_landmarks:
      for faceLms in result.multi_face_landmarks:
        mpDraw.draw_landmarks(
          img, 
          faceLms, 
          mp_face_mesh.FACEMESH_CONTOURS,
          baseDot_drawing_spec,
          baseLine_drawing_spec 
          )

    cv2.imshow("img", img)

  if cv2.waitKey(5) & 0xFF == ord("q"):
    break


# from MediaPipe Python Solution API:

# # For webcam input:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image)

#     # Draw the face mesh annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_face_landmarks:
#       for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_iris_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()