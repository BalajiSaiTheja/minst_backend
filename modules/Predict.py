import torch
import cv2
import numpy as np
import os


class Predict:
    model = None
    def __init__(self,path = "./model/model_vit.pt"):
        if Predict.model is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            MODEL_PATH = os.path.join(BASE_DIR, "model", "model_vit.pt")

            print("Loading model from:", MODEL_PATH)

            Predict.model = torch.jit.load(MODEL_PATH, map_location="cpu")
            Predict.model.eval()
    @staticmethod
    def _initialize(path=None):
        if path is not None:
            return Predict(path=path)
        return Predict()
    
    @staticmethod
    def predict(image):
        if Predict.model is None:
            Predict._initialize()
            
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not load image at {image}. Check if file exists.")
            return None
        
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]

        f = 20.0 / max(w, h)
        digit = cv2.resize(digit, (int(w*f), int(h*f)), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        h_d, w_d = digit.shape
        y_off, x_off = (28 - h_d) // 2, (28 - w_d) // 2
        canvas[y_off:y_off+h_d, x_off:x_off+w_d] = digit

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        x = canvas.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0) 
        x = (x - 0.5) / 0.5

        with torch.no_grad():
            out = Predict.model(x)
        pred = torch.argmax(out, dim=1).item()

        #import matplotlib.pyplot as plt
        #plt.imshow(canvas, cmap='gray')
        #plt.title(f"Processed Input - Prediction: {pred}")
        #plt.show()
        
        return pred

    @staticmethod
    def PredictFromBytes(image):
        if Predict.model is None:
            Predict._initialize()

        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error: Could not load image at {image}. Check if file exists.")
            return None
        
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]

        f = 20.0 / max(w, h)
        digit = cv2.resize(digit, (int(w*f), int(h*f)), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        h_d, w_d = digit.shape
        y_off, x_off = (28 - h_d) // 2, (28 - w_d) // 2
        canvas[y_off:y_off+h_d, x_off:x_off+w_d] = digit

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        x = canvas.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0) 
        x = (x - 0.5) / 0.5

        with torch.no_grad():
            out = Predict.model(x)
        pred = torch.argmax(out, dim=1).item()

        #import matplotlib.pyplot as plt
        #plt.imshow(canvas, cmap='gray')
        #plt.title(f"Processed Input - Prediction: {pred}")
        #plt.show()
        
        return pred
    

if(__name__ == "__main__"):
    
    print("predicted : ",Predict.predict("./imgs/3.png"))