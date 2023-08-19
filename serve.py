import torch
from ultralytics import YOLO
import cv2
from dataclasses import dataclass
import time
from utils.functions import plot_results, rescale_bboxes, transform
from utils.datasets import LoadWebcam, LoadVideo
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Config:
    source: str = "assets/walking_resized.mp4"
    view_img: bool = False
    model_type: str = "detr_resnet50"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    skip: int = 1
    yolo: bool = True
    yolo_type = "yolov8n.pt"


class Detector:
    def __init__(self):
        self.config = Config()
        self.device = self.config.device
        if self.config.source == "0":
            logging.info("Using stream from the webcam")
            self.dataset = LoadWebcam()
        else:
            logging.info("Using stream from the video file: " + self.config.source)
            self.dataset = LoadVideo(self.config.source)
        self.start = time.time()
        self.count = 0

    def load_model(self):
        if self.config.yolo:
            if self.config.yolo_type is None or self.config.yolo_type == "":
                raise ValueError("YOLO model type is not specified")
            model = YOLO(self.config.yolo_type)
            logging.info(f"YOLOv8 Inference using {self.config.yolo_type}")
        else:
            if self.config.model_type is None or self.config.model_type == "":
                raise ValueError("DETR model type is not specified")
            model = torch.hub.load(
                "facebookresearch/detr", self.config.model_type, pretrained=True
            ).to(self.device)
            model.eval()
            logging.info(f"DETR Inference using {self.config.model_type}")
        return model

    def detect(self):
        model = self.load_model()
        for img in self.dataset:
            self.count += 1
            if self.count % self.config.skip != 0:
                continue
            if not self.config.yolo:
                im = transform(img).unsqueeze(0).to(self.device)
                outputs = model(im)
                # keep only predictions with 0.7+ confidence
                probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.9
                bboxes_scaled = rescale_bboxes(
                    outputs["pred_boxes"][0, keep].to("cpu"), img.shape[:2]
                )
            else:
                outputs = model(img)
            logging.info(
                f"FPS: {self.count / self.config.skip / (time.time() - self.start)}"
            )
            # print(f"FPS: {self.count / self.skip / (time.time() - self.start)}")
            if self.config.view_img:
                if self.config.yolo:
                    annotated_frame = outputs[0].plot()
                    cv2.imshow("YOLOv8 Inference", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    plot_results(img, probas[keep], bboxes_scaled)
        logging.info("************************* Done *****************************")


if __name__ == "__main__":
    detector = Detector()
    detector.detect()
