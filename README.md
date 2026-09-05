# Product Verification System

A Django-based system for checking whether a product is likely genuine or
counterfeit, using free AI vision models (via [OpenRouter](https://openrouter.ai))
for image analysis and free public product databases for barcode/QR lookups.
No custom model training or paid AI subscription required.

## Features

- **Image verification** - upload a photo or snap one with your camera; a free
  vision-capable LLM inspects packaging, print quality, logos, and any visible
  security features, and returns an authenticity verdict with a confidence score.
- **Barcode / QR verification** - scan live via the browser's camera, upload a
  photo containing a barcode, or type the number in by hand. Codes are checked
  against [Open Food Facts](https://world.openfoodfacts.org/) and
  [UPCitemdb](https://www.upcitemdb.com/) (both free, no API key needed).
- **Combined verification** - when both an image and a barcode are available,
  results are combined, leaning towards the more cautious verdict if they disagree.
- **Persisted history** - every verification is saved to the database and
  browsable via the History page and the Django admin.
- **Works across product categories** - not limited to one type of product;
  the AI prompt is written to handle electronics, cosmetics, food, drinks,
  clothing, pharmaceuticals, and more.

## Setup

1. Clone the repository and `cd` into it.
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Barcode scanning uses `pyzbar`, which needs the system `zbar` library:
   - macOS: `brew install zbar`
   - Debian/Ubuntu: `sudo apt-get install libzbar0`
   - Windows: the required DLL ships with the `pyzbar` wheel, no extra step needed.

   If `zbar` isn't available, the app still works - it automatically falls
   back to OpenCV's built-in QR code detector (QR only, no linear barcodes).
4. Copy the example environment file and customize it:
   ```
   cp .env.example .env
   ```
5. Get a **free** OpenRouter API key at https://openrouter.ai/keys and set
   `OPENROUTER_API_KEY` in `.env`. This is the only key you need - no billing
   required for the free models this app uses.
6. Apply migrations:
   ```
   python manage.py migrate
   ```
7. (Optional) create an admin account: `python manage.py createsuperuser`

## Running the Application

Development:
```
python manage.py runserver
```
Visit http://127.0.0.1:8000/

Production (example):
```
gunicorn product_verification.wsgi:application
```

A small convenience wrapper is also included:
```
python run.py dev   # same as manage.py runserver
python run.py prod  # same as the gunicorn command above
```

## Environment Variables

See `.env.example` for the full list, including:

- `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `ALLOWED_HOSTS`
- `DATABASE_URL` - leave empty for local SQLite, or set a Postgres URL for production
- `CORS_ALLOWED_ORIGINS`, `CORS_ALLOW_ALL_ORIGINS`
- `OPENROUTER_API_KEY` - **required** for AI image verification
- `OPENROUTER_VISION_MODELS` - optional override of the free-model fallback list

## How verification works

1. `products/ai_agents/image_agent.py` sends the photo to a free vision model
   on OpenRouter (currently `google/gemma-4-31b-it:free`, with automatic
   fallback to a few other free vision models if one is rate-limited) and asks
   for a structured JSON verdict.
2. `products/ai_agents/barcode_agent.py` scans for barcodes/QR codes locally
   (no AI needed) and, for numeric barcodes, checks them against Open Food
   Facts and UPCitemdb.
3. `products/verification_service.py` combines both results into a single
   response, weighting image analysis more heavily since it can actually
   inspect the physical product.
4. `products/views.py` persists the result as a `Verification` (and its
   related `Product`) so it shows up in `/history/` and the Django admin.

## API

- `POST /api/verify/` - form fields: `image` (file, optional), `product_name`
  (optional), `barcode_value` (optional). At least one of `image` or
  `barcode_value` is required.
- `POST /api/verify/barcode/` - form fields: `barcode_value` (required),
  `product_name` (optional). For barcode-only checks with no photo.

Both return JSON with `status`, `confidence`, `message`, `product_details`,
and (for image analysis) `security_features_observed`, `red_flags`, and
`recommendation`.

## Deployment

See `DEPLOYMENT.md` for Render-specific instructions. In short:

1. Set `DJANGO_DEBUG=False` and a real `DJANGO_SECRET_KEY`.
2. Set `ALLOWED_HOSTS` to your real domain(s).
3. Set `OPENROUTER_API_KEY`.
4. Attach a persistent database (Postgres) via `DATABASE_URL` if you need
   verification history to survive redeploys - the default SQLite file is
   fine for local development but isn't persistent on most PaaS hosts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
