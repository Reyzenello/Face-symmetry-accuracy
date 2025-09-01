import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def calculate_symmetry(keypoints, image_width, image_height):
    left_eye = keypoints[0]
    right_eye = keypoints[1]
    nose_tip = keypoints[2]
    mouth_center = keypoints[3]
    left_ear = keypoints[4]
    right_ear = keypoints[5]

    # Convert normalized coordinates to pixel coordinates
    left_eye_px = normalized_to_pixel_coordinates(left_eye.x, left_eye.y, image_width, image_height)
    right_eye_px = normalized_to_pixel_coordinates(right_eye.x, right_eye.y, image_width, image_height)
    nose_tip_px = normalized_to_pixel_coordinates(nose_tip.x, nose_tip.y, image_width, image_height)
    mouth_center_px = normalized_to_pixel_coordinates(mouth_center.x, mouth_center.y, image_width, image_height)
    left_ear_px = normalized_to_pixel_coordinates(left_ear.x, left_ear.y, image_width, image_height)
    right_ear_px = normalized_to_pixel_coordinates(right_ear.x, right_ear.y, image_width, image_height)

    # Calculate midpoint between eyes
    midpoint_x = (left_eye_px[0] + right_eye_px[0]) / 2

    # Calculate symmetry scores (0 to 1, where 1 is perfectly symmetrical)
    eye_symmetry = 1 - abs(left_eye_px[1] - right_eye_px[1]) / image_height
    nose_symmetry = 1 - abs(nose_tip_px[0] - midpoint_x) / image_width
    mouth_symmetry = 1 - abs(mouth_center_px[0] - midpoint_x) / image_width
    ear_symmetry = 1 - abs((left_ear_px[0] - midpoint_x) - (midpoint_x - right_ear_px[0])) / image_width

    # Calculate overall symmetry (weighted average)
    overall_symmetry = (eye_symmetry * 0.3 + nose_symmetry * 0.2 + mouth_symmetry * 0.3 + ear_symmetry * 0.2)

    return {
        "eyes": eye_symmetry * 100,
        "nose": nose_symmetry * 100,
        "mouth": mouth_symmetry * 100,
        "ears": ear_symmetry * 100,
        "overall": overall_symmetry * 100
    }

def visualize(image, detection_result, symmetry_scores):
    annotated_image = image.copy()
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)

        for idx, keypoint in enumerate(detection.keypoints):
            kp_px = normalized_to_pixel_coordinates(keypoint.x, keypoint.y, annotated_image.shape[1], annotated_image.shape[0])
            cv2.circle(annotated_image, kp_px, 5, (255, 0, 0), -1)

        y_offset = bbox.origin_y - 10
        for feature, score in symmetry_scores.items():
            text = f"{feature.capitalize()} Symmetry: {score:.1f}%"
            cv2.putText(annotated_image, text, (bbox.origin_x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 20

    return annotated_image

def main():
    IMAGE_FILE = r'<file.jpg>'
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    image = mp.Image.create_from_file(IMAGE_FILE)
    detection_result = detector.detect(image)

    if not detection_result.detections:
        print("No faces detected in the image.")
        return

    image_copy = np.copy(image.numpy_view())
    symmetry_scores = calculate_symmetry(detection_result.detections[0].keypoints, 
                                         image_copy.shape[1], image_copy.shape[0])

    print("Symmetry Scores:")
    for feature, score in symmetry_scores.items():
        print(f"{feature.capitalize()}: {score:.1f}%")

    annotated_image = visualize(image_copy, detection_result, symmetry_scores)
    cv2.imshow('Face Symmetry Analysis', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
