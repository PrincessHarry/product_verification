#!/usr/bin/env bash
# exit on error
set -o errexit

# Barcode scanning (pyzbar) needs the system zbar shared library. Render's
# Python environment allows apt-get during build; this is safe to skip if
# apt isn't available (e.g. running this script somewhere else).
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y && apt-get install -y --no-install-recommends libzbar0 || true
fi

# Install Python dependencies
pip install -r requirements.txt

# Run Django migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --no-input
