from django.core.management.base import BaseCommand
from products.collect_initial_data import main as collect_data_main
import sys

class Command(BaseCommand):
    help = 'Collect initial data for product verification'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting data collection...'))
        
        # Run the data collection script
        try:
            collect_data_main()
            self.stdout.write(self.style.SUCCESS('Data collection completed successfully!'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during data collection: {str(e)}'))
            sys.exit(1) 