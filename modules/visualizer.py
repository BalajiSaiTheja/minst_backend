import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from modules.Predict import Predict
from modules.ViT import ViT
import io
import os

class Visualize:
    model = None
    def __init__(self,path="./model/best_model_vit.pth"):
        if Visualize.model is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_vit.pth")
            print("Loading ViT model from:", MODEL_PATH)
            Visualize.model = ViT()
            Visualize.model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu")
            )
            Visualize.model.eval()
            
    @staticmethod
    def _initialize(path=None):
        if path is not None:
            return Visualize(path=path)
        return Visualize()
    
    @staticmethod
    def visualize_attention(image):
        if Predict.model is None:
            Predict._initialize()

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(img)
        x_, y_, w, h = cv2.boundingRect(coords)
        digit = img[y_:y_+h, x_:x_+w]

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
        x.requires_grad_()
        out = Predict.model(x)
        pred_class = out.argmax()
        out[0, pred_class].backward()
        saliency = x.grad.abs().squeeze().cpu().numpy()
        return canvas, saliency, pred_class.item()
    
    @staticmethod
    def get_layer_activations(image_path,buffer=False):
        if Visualize.model is None:
            Visualize._initialize()

        model = Visualize.model

        if buffer:
            nparr = np.frombuffer(image_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(img)
        x_, y_, w, h = cv2.boundingRect(coords)
        digit = img[y_:y_+h, x_:x_+w]

        f = 20.0 / max(w, h)
        digit = cv2.resize(digit, (int(w*f), int(h*f)))

        canvas = np.zeros((28, 28), dtype=np.uint8)
        h_d, w_d = digit.shape
        y_off, x_off = (28 - h_d) // 2, (28 - w_d) // 2
        canvas[y_off:y_off+h_d, x_off:x_off+w_d] = digit
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        x = torch.from_numpy(canvas.astype(np.float32) / 255.0).view(1, 1, 28, 28)
        x = (x - 0.5) / 0.5

        activations = {
            "attn": [],
            "mlp": [],
            "block": []
        }
        hooks = []

        def attn_hook(module, input, output):
            activations["attn"].append(output.detach())

        def mlp_hook(module, input, output):
            activations["mlp"].append(output.detach())

        def block_hook(module, input, output):
            activations["block"].append(output.detach())

        for enc in model.encoder:
            hooks.append(enc.mha.register_forward_hook(attn_hook))
            hooks.append(enc.mlp.register_forward_hook(mlp_hook))
            hooks.append(enc.register_forward_hook(block_hook))

        with torch.no_grad():
            _ = model(x)

        for h in hooks:
            h.remove()

        return canvas, activations

    @staticmethod
    def plot_layers(image_path, buffer=False):
        canvas, acts = Visualize.get_layer_activations(image_path, buffer=buffer)

        attn_layers = acts["attn"]
        mlp_layers = acts["mlp"]
        block_layers = acts["block"]
        num_layers = min(len(block_layers), 6)
        fig, axes = plt.subplots(3, num_layers, figsize=(num_layers * 3, 10))

        def process_and_plot(layer_list, row_idx, cmap, row_label):
            for i, act in enumerate(layer_list):
                patch_tokens = act[0, 1:, :]
                heatmap = patch_tokens.norm(dim=-1)
                grid = int(len(heatmap) ** 0.5)
                heatmap = heatmap.view(grid, grid).cpu().numpy()

                ax = axes[row_idx, i]
                im = ax.imshow(heatmap, cmap=cmap)
                if i == 0:
                    ax.set_ylabel(row_label, fontsize=14, fontweight='bold')
                
                ax.set_title(f"Layer {i}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        process_and_plot(attn_layers, 0, 'Blues', 'Attention')
        process_and_plot(mlp_layers, 1, 'Reds', 'MLP')
        process_and_plot(block_layers, 2, 'viridis', 'Blocks')

        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
        
        if buffer:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            img_buf.seek(0)
            return img_buf.getvalue()
        
        plt.show()
    
if __name__ == "__main__":
    canvas, saliency, pred = Visualize.visualize_attention("./imgs/9.png")

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(canvas, cmap='gray')
    plt.title(f"Input (Pred: {pred})")
    plt.subplot(1,2,2)
    plt.imshow(saliency, cmap='hot')
    plt.title("Attention (Saliency)")
    plt.show()

    Visualize.plot_layers("./imgs/3.png")