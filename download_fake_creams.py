import os
import requests
from PIL import Image
from io import BytesIO
import time

def download_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.save(save_path, 'JPEG')
            print(f"Successfully downloaded: {save_path}")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    return False

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Nigerian cream brands and their fake product image URLs
brands = {
    "Fair_&_White": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Pure_White": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Bio_Claire": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Caro_White": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "QEI+": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Perfect_White": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Rapid_White": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Skin_Light": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Movate": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Crusader": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Tura": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Venus": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Clear_Essence": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Fashion_Fair": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    },
    "Makari": {
        "fake": [
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/~ZwAAOSwPLNjwL6Y/s-l1600.jpg"
        ]
    }
}

def main():
    base_path = "dataset/images/Cosmetics/Cream"
    
    for brand, data in brands.items():
        brand_path = os.path.join(base_path, brand)
        fake_path = os.path.join(brand_path, "fake")
        
        # Create directories if they don't exist
        create_directory(brand_path)
        create_directory(fake_path)
        
        # Download fake images
        for i, url in enumerate(data["fake"], 1):
            save_path = os.path.join(fake_path, f"fake_{i}.jpg")
            if not os.path.exists(save_path):
                download_image(url, save_path)
                time.sleep(1)  # Add delay between downloads

if __name__ == "__main__":
    main() 