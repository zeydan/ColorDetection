import cv2
import numpy as np
import json

class ColorDetection:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.config = self.load_config()
        self.create_trackbars(self.config)

    @staticmethod
    def load_config():
        try:
            with open('config.json', 'r') as file:
                config = json.load(file)
            return config
        except:
            config = {
                'Hue Lower' : 0,
                'Hue Upper' : 0,
                'Sat Lower' : 0,
                'Sat Upper' : 0,
                'Val Lower' : 0,
                'Val Upper' : 0
            }
            return config

    @staticmethod
    def create_trackbars(config):
        cv2.namedWindow('TrackBars', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('TrackBars', 1000, 10)
        for key, value in config.items():
            cv2.createTrackbar(key, 'TrackBars', value, 255, empty)

    def start_loop(self):
        while True:
            _, self.frame = self.cap.read()

            # Get color samples from trackbar or mouse click and save them
            cv2.setMouseCallback('Frame', self.set_trackbars)
            self.config = self.get_trackbars(self.config)
            self.save_config(self.config)

            lowerLimit, upperLimit = self.get_limits(self.config)
            hsvFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            self.mask = cv2.inRange(hsvFrame, lowerLimit, upperLimit)
            self.draw_border()
            
            cv2.imshow('Mask', self.mask)
            cv2.imshow('Frame', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def set_trackbars(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = self.frame[y][x]
            color = np.uint8([[color]])
            hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

            self.config = {
                'Hue Lower' : max(hsvColor[0] - 10, 0),
                'Hue Upper' : min(hsvColor[0] + 10, 255),
                'Sat Lower' : max(hsvColor[1] - 50, 0),
                'Sat Upper' : min(hsvColor[1] + 50, 255),
                'Val Lower' : max(hsvColor[2] - 50, 0),
                'Val Upper' : min(hsvColor[2] + 50, 255)
            }
            
            for key, value in self.config.items():
                cv2.setTrackbarPos(key, 'TrackBars', value)

    @staticmethod
    def get_trackbars(config):
        for key in config.keys():
            config[key] = cv2.getTrackbarPos(key, 'TrackBars')
        return config

    @staticmethod
    def save_config(config):
        with open('config.json', 'w') as file:
            file.write(json.dumps(config, indent=4))

    @staticmethod
    def get_limits(config):
        lowerLimit = config['Hue Lower'], config['Sat Lower'], config['Val Lower']
        upperLimit = config['Hue Upper'], config['Sat Upper'], config['Val Upper']
        lowerLimit = np.array(lowerLimit, dtype=np.uint8)
        upperLimit = np.array(upperLimit, dtype=np.uint8)
        return lowerLimit, upperLimit

    def draw_border(self):
        contours, _= cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 2)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

# For trackbars
def empty(x):
    pass

if __name__ == '__main__':
    detector = ColorDetection()
    detector.start_loop()