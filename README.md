# detr-vision
Implementation of DETR: End-to-End Object Detection with Transformers in PyTorch. In this repo, I have used DETR model
and yolo models on real-time video stream. Currently, Yolo is commonly used in real-time object detection application.
DETR is a new model architecture (shown bellow) for object detection that uses transformers.

![img](assets/detr.png)

The server.py uses datasets from webcam / ip camera or video file. This information can be changed in server.py dataclass.

```python
@dataclass
class Config:
    source: str = "assets/walking_resized.mp4"
    view_img: bool = False
    model_type: str = "detr_resnet50"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    skip: int = 1
    yolo: bool = True
    yolo_type = "yolov8n.pt"
```

## How to run
First, install the requirements:
```bash
pip install -r requirements.txt
```
Then, run the server:
```bash
python server.py
```

## How to run with docker
First, build the docker image:
```bash
docker build -t detr-vision .
```
Then, run the docker container:
```bash
docker run -it --rm --name detr-vision-container detr-vision
```

Run with docker-compose:
```bash
docker-compose up --build
```
stop the container with docker-compose:
```bash
docker-compose down
```
