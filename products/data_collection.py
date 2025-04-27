import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import json
import logging
from django.conf import settings
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.videos_dir = os.path.join(data_dir, "videos")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = self._load_metadata()
        
        # Initialize Gemini AI
        self.gemini = self._initialize_gemini()
        
        # Predefined image URLs for common products
        self.predefined_images = {
            "Samsung Galaxy S21": {
                "original": [
                    "https://images.samsung.com/is/image/samsung/p6pim/uk/galaxy-s21/gallery/uk-galaxy-s21-5g-g991-sm-g991bzadeub-thumb-368338803",
                    "https://images.samsung.com/is/image/samsung/p6pim/uk/galaxy-s21/gallery/uk-galaxy-s21-5g-g991-sm-g991bzadeub-368338803",
                    "https://images.samsung.com/is/image/samsung/p6pim/uk/galaxy-s21/gallery/uk-galaxy-s21-5g-g991-sm-g991bzadeub-368338804",
                    "https://images.samsung.com/is/image/samsung/p6pim/uk/galaxy-s21/gallery/uk-galaxy-s21-5g-g991-sm-g991bzadeub-368338805",
                    "https://images.samsung.com/is/image/samsung/p6pim/uk/galaxy-s21/gallery/uk-galaxy-s21-5g-g991-sm-g991bzadeub-368338806"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                ]
            },
            "iPhone 13": {
                "original": [
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-blue-select-2021?wid=940&hei=1112&fmt=png-alpha&.v=1645572386470",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-mini-blue-select-2021?wid=940&hei=1112&fmt=png-alpha&.v=1645572386470",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-pro-max-graphite-select?wid=940&hei=1112&fmt=png-alpha&.v=1645552346280",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-pro-sierra-blue-select?wid=940&hei=1112&fmt=png-alpha&.v=1645552346280",
                    "https://store.storeimages.cdn-apple.com/4982/as-images.apple.com/is/iphone-13-pro-max-sierra-blue-select?wid=940&hei=1112&fmt=png-alpha&.v=1645552346280"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                ]
            },
            "Nike Air Max": {
                "original": [
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png",
                    "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/skwgyqrbfzhu6uyeh0gg/air-max-270-shoes-V4DfZQ.png"
                ],
                "fake": [
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg",
                    "https://i.ebayimg.com/images/g/0~AAAOSw~1Fj~y~Y/s-l1600.jpg"
                ]
            }
        }
        
        self.categories = {
            "Category": {
                "Subcategory": {
                    "brands": {
                        "Brand_Name": {
                            "original": ["url1", "url2"],
                            "fake": ["url1", "url2"]
                        }
                    }
                }
            }
        }
    
    def _load_metadata(self):
        """Load metadata from file or create new if not exists"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "products": [],
            "videos": []
        }
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def collect_product_data(self, product_name, barcode, original_image_url, fake_image_urls):
        """Collect data for a specific product"""
        product_id = f"{product_name}_{barcode}"
        product_dir = os.path.join(self.images_dir, product_id)
        os.makedirs(product_dir, exist_ok=True)
        
        # Download original image
        original_path = os.path.join(product_dir, "original.jpg")
        self._download_image(original_image_url, original_path)
        
        # Download fake images
        fake_paths = []
        for i, url in enumerate(fake_image_urls):
            fake_path = os.path.join(product_dir, f"fake_{i+1}.jpg")
            self._download_image(url, fake_path)
            fake_paths.append(fake_path)
        
        # Add to metadata
        product_data = {
            "id": product_id,
            "name": product_name,
            "barcode": barcode,
            "original_image": original_path,
            "fake_images": fake_paths,
            "features": self._extract_features(original_path)
        }
        
        self.metadata["products"].append(product_data)
        self._save_metadata()
        
        return product_data
    
    def collect_youtube_videos(self, search_query, max_videos=10):
        """Collect YouTube videos about product verification"""
        try:
            from pytube import Search
            
            search = Search(search_query)
            videos = []
            
            for i, video in enumerate(search.results):
                if i >= max_videos:
                    break
                    
                video_path = os.path.join(self.videos_dir, f"{video.video_id}.mp4")
                
                # Download video
                video.streams.filter(progressive=True, file_extension='mp4').first().download(
                    output_path=self.videos_dir,
                    filename=f"{video.video_id}.mp4"
                )
                
                # Extract frames
                frames = self._extract_frames(video_path)
                
                # Add to metadata
                video_data = {
                    "id": video.video_id,
                    "title": video.title,
                    "url": video.watch_url,
                    "path": video_path,
                    "frames": frames
                }
                
                videos.append(video_data)
            
            self.metadata["videos"].extend(videos)
            self._save_metadata()
            
            return videos
        except ImportError:
            logger.error("pytube package not installed. Please install it with: pip install pytube")
            return []
    
    def _download_image(self, url, save_path):
        """Download an image from URL"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                logger.info(f"Downloaded image to {save_path}")
            else:
                logger.error(f"Failed to download image from {url}. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
    
    def _extract_features(self, image_path):
        """Extract features from an image for comparison"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return {}
                
            # Resize for consistency
            img = cv2.resize(img, (224, 224))
            
            # Extract color histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Extract texture features using Gabor filter
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            texture_features = []
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                texture_features.extend([np.mean(filtered), np.std(filtered)])
            
            return {
                "color_histogram": hist.tolist(),
                "texture_features": texture_features
            }
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return {}
    
    def _extract_frames(self, video_path, frame_interval=30):
        """Extract frames from a video at regular intervals"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(os.path.dirname(video_path), 
                                             f"{os.path.basename(video_path)}_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)
                    
                frame_count += 1
                
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return []
    
    def add_product_to_database(self, product_data):
        """Add a product to the database"""
        from .models import Product, Verification
        
        try:
            # Create or update product
            product, created = Product.objects.get_or_create(
                product_code=product_data["barcode"],
                defaults={
                    'name': product_data["name"],
                    'manufacturer': product_data.get("manufacturer", "Unknown"),
                    'description': product_data.get("description", ""),
                    'barcode': product_data["barcode"],
                    'manufacturing_date': product_data.get("manufacturing_date", "2023-01-01"),
                    'batch_number': product_data.get("batch_number", str(uuid.uuid4())),
                    'verification_status': 'verified'
                }
            )
            
            if not created:
                # Update existing product
                product.name = product_data["name"]
                product.manufacturer = product_data.get("manufacturer", product.manufacturer)
                product.description = product_data.get("description", product.description)
                product.verification_status = 'verified'
                product.save()
            
            # Create verification record
            verification = Verification.objects.create(
                product=product,
                verification_method='dataset',
                status='success',
                confidence=1.0,
                details=json.dumps(product_data),
                verification_id=str(uuid.uuid4()),
                metadata={"source": "dataset"}
            )
            
            logger.info(f"Added product {product.name} to database")
            return product, verification
        except Exception as e:
            logger.error(f"Error adding product to database: {str(e)}")
            return None, None

    def collect_product_images(self, product_name, max_images=5):
        """Collect product images from the internet"""
        try:
            # Parse brand and product name
            parts = product_name.split(' ', 1)
            if len(parts) != 2:
                logger.error(f"Invalid product name format: {product_name}")
                return
            
            brand, product = parts
            
            # Create brand directory
            brand_dir = os.path.join(self.images_dir, brand.replace(" ", "_"))
            os.makedirs(brand_dir, exist_ok=True)
            
            # Create product directory
            product_dir = os.path.join(brand_dir, product.replace(" ", "_"))
            product_original_dir = os.path.join(product_dir, "original")
            product_fake_dir = os.path.join(product_dir, "fake")
            
            os.makedirs(product_original_dir, exist_ok=True)
            os.makedirs(product_fake_dir, exist_ok=True)
            
            # Search for original product images
            logger.info(f"Searching for original {product_name} images...")
            original_urls = self._search_product_images(product_name, is_fake=False)
            
            # Search for fake product images
            logger.info(f"Searching for fake {product_name} images...")
            fake_urls = self._search_product_images(product_name, is_fake=True)
            
            # Download original images
            for i, url in enumerate(original_urls[:max_images]):
                try:
                    image_path = os.path.join(product_original_dir, f"original_{i+1}.jpg")
                    self._download_image(url, image_path)
                    logger.info(f"Downloaded original image {i+1} for {product_name}")
                except Exception as e:
                    logger.error(f"Error downloading original image {i+1}: {str(e)}")
            
            # Download fake images
            for i, url in enumerate(fake_urls[:max_images]):
                try:
                    image_path = os.path.join(product_fake_dir, f"fake_{i+1}.jpg")
                    self._download_image(url, image_path)
                    logger.info(f"Downloaded fake image {i+1} for {product_name}")
                except Exception as e:
                    logger.error(f"Error downloading fake image {i+1}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error collecting images for {product_name}: {str(e)}")
            raise
            
    def _search_product_images(self, product_name, is_fake=False):
        """Search for product images using multiple sources"""
        try:
            # Create search query
            search_query = f"{product_name} {'fake' if is_fake else 'original'} product image"
            
            # List of image URLs to return
            urls = []
            
            # Method 1: Use predefined image URLs for common products
            if product_name in self.predefined_images:
                urls = self.predefined_images[product_name]["fake" if is_fake else "original"]
                logger.info(f"Found {len(urls)} predefined images for {product_name}")
            
            # Method 2: Use web scraping to find images
            if not urls:
                scraped_urls = self._scrape_image_urls(search_query)
                if scraped_urls:
                    urls.extend(scraped_urls)
                    logger.info(f"Found {len(scraped_urls)} scraped images for {product_name}")
            
            # Method 3: Use Gemini AI as a fallback
            if not urls:
                try:
                    response = self.gemini.generate_content(search_query)
                    for line in response.text.split('\n'):
                        if line.startswith('http') and ('.jpg' in line or '.png' in line):
                            urls.append(line.strip())
                    logger.info(f"Found {len(urls)} images from Gemini AI for {product_name}")
                except Exception as e:
                    logger.error(f"Error using Gemini AI: {str(e)}")
            
            # Remove duplicates
            urls = list(set(urls))
            
            return urls
            
        except Exception as e:
            logger.error(f"Error searching for images: {str(e)}")
            return []
            
    def _scrape_image_urls(self, search_query):
        """Scrape image URLs from search results"""
        try:
            # Use a search engine to find images
            search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs
            urls = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src.startswith('http') and ('.jpg' in src or '.png' in src):
                    urls.append(src)
            
            return urls
            
        except Exception as e:
            logger.error(f"Error scraping image URLs: {str(e)}")
            return []
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            import google.generativeai as genai
            
            # Get API key from environment variable
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found in environment variables. Using default key.")
                api_key = "AIzaSyDXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your actual API key
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Get the model
            model = genai.GenerativeModel('gemini-pro')
            
            logger.info("Gemini AI initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing Gemini AI: {str(e)}")
            return None 