import cv2


class LoadWebcam:
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        if pipe.isnumeric():
            pipe = eval(pipe)

        self.pipe = pipe
        self.cap = cv2.VideoCapture(self.pipe)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        if self.pipe == 0:
            ret, frame = self.cap.read()
            height, width = frame.shape[:2]
            ratio = self.img_size / max(height, width)
            frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
            if not ret:
                raise StopIteration

        return frame

    def __len__(self):
        return 0


class LoadVideo:
    def __init__(self, file_path, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.cap = cv2.VideoCapture(file_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        ret, frame = self.cap.read()
        height, width = frame.shape[:2]
        ratio = self.img_size / max(height, width)
        frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
        if not ret:
            raise StopIteration
        return frame

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
