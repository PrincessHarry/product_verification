# Product Verification System

This system uses deep learning to verify if a product is original or fake based on image analysis.

## Overview

The product verification system consists of several components:

1. **Data Collection**: Scripts to collect and organize product images
2. **Model Training**: Script to train a deep learning model on the collected data
3. **Product Verification**: Script to verify if a product is original or fake

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch torchvision pillow numpy scikit-learn
   ```

## Usage

### Verifying a Product

To verify if a product is original or fake, use the `verify_product.py` script:

```
python verify_product.py path/to/image.jpg
```

Example:
```
python verify_product.py "dataset/images/Cosmetics/Cream/Fair_&_White/original/original_1.jpg"
```

The script will output:
- The verification result (ORIGINAL or FAKE)
- Confidence score
- Probabilities for original and fake

### Loading the Model

If you need to load the model in your own code, use the `load_model.py` script:

```python
from load_model import load_model

# Load the model
model = load_model(model_path="dataset/product_verification_model.pth", model_type="mobilenet")
```

## Model Architecture

The system uses a MobileNetV2 model with a custom classifier for binary classification (original vs. fake). The model is trained on a dataset of original and fake product images.

## Dataset Structure

The dataset is organized as follows:

```
dataset/
├── images/
│   ├── Cosmetics/
│   │   ├── Cream/
│   │   │   ├── Brand_Name/
│   │   │   │   ├── original/
│   │   │   │   └── fake/
│   │   ├── Skincare/
│   │   └── Soap/
│   ├── Electronics/
│   └── Fashion/
├── metadata.json
└── product_verification_model.pth
```

## Training the Model

To train the model on your own dataset, use the `train_model.py` script:

```
python train_model.py --model mobilenet --epochs 10 --batch_size 16 --lr 0.001
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 