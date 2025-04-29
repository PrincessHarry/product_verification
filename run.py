#!/usr/bin/env python
"""
Simple script to run the Django server in either development or production mode.
Usage:
    python run.py dev     # Run in development mode
    python run.py prod    # Run in production mode
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def run_server(mode):
    # Load environment variables
    load_dotenv()
    
    # Set environment variables based on mode
    if mode == 'dev':
        os.environ['DJANGO_DEBUG'] = 'True'
        os.environ['DJANGO_SETTINGS_MODULE'] = 'product_verification.settings'
        print("Starting development server...")
        # Run with HTTPS in development
        subprocess.run(['python', 'manage.py', 'runserver_plus', '--cert-file', 'cert.crt', '--key-file', 'cert.key', '0.0.0.0:8000'])
    elif mode == 'prod':
        os.environ['DJANGO_DEBUG'] = 'False'
        os.environ['DJANGO_SETTINGS_MODULE'] = 'product_verification.settings'
        print("Starting production server...")
        subprocess.run(['gunicorn', 'product_verification.wsgi:application', '--bind', '0.0.0.0:8000'])
    else:
        print("Invalid mode. Use 'dev' or 'prod'")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run.py [dev|prod]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    run_server(mode) 