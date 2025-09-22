import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==== 匯入自定義模組 ====
from model.autoencoder import Autoencoder
from model.opticalSimulation import ONN
from model.restormer250724 import Restormer
from model.sensor import Sensor, SensorNoise
from config import ENCODER_CONFIG, SENSOR_CONFIG, RESTORMER_CONFIG, AUTOENCODER_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 影像預處理 ====
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 跟訓練一致
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

# ==== 建立模型 ====
def build_model():
    encoder = ONN(ENCODER_CONFIG).to(device)
    sensor = Sensor(SENSOR_CONFIG).to(device)
    sensor_noise = SensorNoise(SENSOR_CONFIG)
    decoder = Restormer(RESTORMER_CONFIG).to(device)  # 雖然不用 decoder，但 Autoencoder 需要

    model = Autoencoder(
        encoder=encoder, 
        decoder=decoder, 
        sensor=sensor, 
        sensor_noise=sensor_noise, 
        config=AUTOENCODER_CONFIG
    ).to(device)
    return model

# ==== 載入模型權重 ====
def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==== 單張推論 ====
def inference_single(model, image_path, save_dir="./inference_results"):
    os.makedirs(save_dir, exist_ok=True)

    # load + preprocess
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 1, H, W]

    with torch.no_grad():
        encoded = model.encoder(img_tensor)
        sensed = model.sensor(encoded)

    # 存 tensor
    tensor_path = os.path.join(save_dir, os.path.basename(image_path).replace(".png", "_sensed.pt"))
    torch.save(sensed.cpu(), tensor_path)

    # 存圖片
    sensed_img = sensed.squeeze().cpu().numpy()
    plt.imshow(sensed_img, cmap="gray")
    plt.axis("off")
    img_path = os.path.join(save_dir, os.path.basename(image_path).replace(".png", "_sensed.png"))
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved results: {tensor_path}, {img_path}")

# ==== 視覺化工具 ====
def plot_field(field, title_prefix="", save_path=None):
    magnitude = torch.abs(field).detach().cpu().numpy()
    phase = torch.angle(field).detach().cpu().numpy()

    plt.figure(figsize=(10, 4))

    # Magnitude
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude, cmap='gray')
    plt.title(f"{title_prefix} Magnitude")
    plt.colorbar()

    # Phase
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='twilight')
    plt.title(f"{title_prefix} Phase")
    plt.colorbar()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"已儲存圖片至 {save_path}")

    plt.show()

# ==== 主程式 ====
if __name__ == "__main__":
    """
    image_dir = "./data/GD_processed"   # 假設這裡有 001.png, 002.png ...

    model = build_model()
    print(model)

    # 推論前 10 張 (001.png ~ 010.png)
    for i in range(1, 11):
        image_path = os.path.join(image_dir, f"{i:03d}.png")
        if os.path.exists(image_path):
            inference_single(model, image_path)
        else:
            print(f"Warning: {image_path} not found.")"""
    num_size = 128
    input_field = torch.zeros((num_size, num_size), dtype=torch.cfloat).to(device)
    input_field[num_size//2, num_size//2] = 1.0 + 0j  # 中央點光源

    model = build_model()
    output = model(input_field)

    print("輸出張量大小:", output.shape)
    os.makedirs("./inference_results", exist_ok=True)
    plot_field(input_field, title_prefix="Input", save_path="./inference_results/Input.png")
    plot_field(output, title_prefix="Output", save_path="./inference_results/Output_diffraction.png")
    
