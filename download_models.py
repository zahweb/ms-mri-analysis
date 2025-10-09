import gdown
import os

def download_unet_model():
    """Download U-Net model from Google Drive"""
    if not os.path.exists("best_unet_final.keras"):
        print("📥 Downloading U-Net model from Google Drive...")
        url = "https://drive.google.com/uc?id=1CgugA_Ti0prkQH3j7NL_pEmXjZx-FfdB"
        gdown.download(url, "best_unet_final.keras", quiet=False)
        print("✅ U-Net model downloaded successfully")
    else:
        print("✅ U-Net model already exists")

if __name__ == "__main__":
    download_unet_model()
