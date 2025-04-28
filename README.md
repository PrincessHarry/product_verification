# Product Verification System

A Django-based system for verifying product authenticity using image analysis and barcode scanning.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy the example environment file and customize it:
   ```
   cp .env.example .env
   ```
5. Edit the `.env` file with your specific settings

## Running the Application

### Development Mode

For local development with HTTP:

```
python run.py dev
```

This will start the Django development server at http://127.0.0.1:8000/

### Production Mode

For production deployment:

```
python run.py prod
```

This will start the Gunicorn server at http://0.0.0.0:8000/

## Environment Variables

The following environment variables can be configured in the `.env` file:

- `DJANGO_SECRET_KEY`: Secret key for Django
- `DJANGO_DEBUG`: Set to 'True' for development, 'False' for production
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts
- `DATABASE_URL`: Database connection URL (leave empty for SQLite in development)
- `CORS_ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `FILE_UPLOAD_MAX_MEMORY_SIZE`: Maximum file upload size in bytes
- `DATA_UPLOAD_MAX_MEMORY_SIZE`: Maximum data upload size in bytes

## Deployment

For production deployment:

1. Set `DJANGO_DEBUG=False` in your `.env` file
2. Configure a proper database URL in `DATABASE_URL`
3. Set up a web server like Nginx to proxy requests to the application
4. Run the application with `python run.py prod`

## Features

- Product verification using image analysis
- Barcode scanning
- Real-time verification results
- Secure database of verified products

## Overview

The product verification system consists of several components:

1. **Data Collection**: Scripts to collect and organize product images
2. **Model Training**: Script to train a deep learning model on the collected data
3. **Product Verification**: Script to verify if a product is original or fake

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