import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "FISH COUNT")
    parser.add_argument(
        "--webcam-resolution", 
        default = [640, 480], 
        nargs = 2, 
        type = int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        
        thickness = 1,
        text_thickness = 1,
        text_scale = 0.3,
        text_color=sv.Color.black(),
        color=sv.Color.green()
        
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone = zone, 
        color = sv.Color.white(),
        thickness = 1,
        text_thickness = 1,
        text_scale = 1
    )

    while True:
        ret, frame = cap.read()

        cv2.imshow("CAMERA", frame)

        key = cv2.waitKey(30)

        if key == ord('s'):
            
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            labels = [
               f"{model.model.names[class_id]} {confidence:0.2f}"
                  for _, confidence, class_id, _
                     in detections
            ]
            frame = box_annotator.annotate(
               scene = frame, 
               detections = detections, 
               #labels = labels
            )

            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)      
            
            cv2.imwrite("result.jpg", frame)

            img = cv2.imread("result.png")
            cv2.waitKey(10)
            
            cv2.imshow('RESULT', frame)
            cv2.waitKey(0)
            
            result = False
        
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()