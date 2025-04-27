from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from .base_agent import BaseVerificationAgent
from products.models import Product

class BarcodeVerificationAgent(BaseVerificationAgent):
    def __init__(self):
        # List of online databases for barcode lookup
        self.databases = [
            'https://www.gs1.org/services/barcode-lookup',
            'https://www.barcodelookup.com',
            'https://www.upcdatabase.com'
        ]
        
        # Initialize barcode verification agent
        super().__init__()
        
        # Update agent metadata
        self.metadata.update({
            'verification_type': 'barcode',
            'databases': self.databases,
            'verification_methods': ['online_lookup', 'local_database']
        })

    async def search_online(self, barcode: str) -> List[Dict[str, Any]]:
        """Search multiple online databases for barcode information"""
        tasks = [self._search_database(db, barcode) for db in self.databases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                continue
            if result and isinstance(result, dict):
                valid_results.append(result)
        
        return valid_results

    async def _search_database(self, database: str, barcode: str) -> Optional[Dict[str, Any]]:
        """Search a specific database for barcode information"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{database}/{barcode}") as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._extract_product_info(html)
        except Exception as e:
            self.add_resource(
                self._create_verification_resource({
                    'error': str(e),
                    'database': database,
                    'barcode': barcode,
                    'status': 'error'
                })
            )
        return None

    def _extract_product_info(self, html: str) -> Dict[str, Any]:
        """Extract product information from HTML response"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Common selectors for product information
        selectors = {
            'name': ['h1', '.product-name', '.title'],
            'manufacturer': ['.manufacturer', '.brand', '.company'],
            'price': ['.price', '.cost', '.amount'],
            'description': ['.description', '.details', '.info']
        }
        
        info = {}
        for field, selector_list in selectors.items():
            for selector in selector_list:
                element = soup.select_one(selector)
                if element:
                    info[field] = element.text.strip()
                    break
        
        return info

    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple sources"""
        if not results:
            return {}
        
        combined = {}
        for result in results:
            for key, value in result.items():
                if key not in combined:
                    combined[key] = value
                elif combined[key] != value:
                    # If values differ, mark as inconsistent
                    combined[key] = f"inconsistent: {combined[key]} vs {value}"
        
        return combined

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on result consistency"""
        if not results:
            return 0.0
        
        # Count consistent fields across results
        consistent_fields = 0
        total_fields = 0
        
        for field in ['name', 'manufacturer', 'price', 'description']:
            values = [r.get(field) for r in results if field in r]
            if values:
                total_fields += 1
                if all(v == values[0] for v in values):
                    consistent_fields += 1
        
        return consistent_fields / total_fields if total_fields > 0 else 0.0

    async def verify_authenticity(self, barcode: str) -> Dict[str, Any]:
        """Verify product authenticity using barcode"""
        try:
            # First check local database
            try:
                product = await Product.objects.filter(barcode=barcode).first()
                if product:
                    result = {
                        "status": "success",
                        "message": "Product found in local database",
                        "confidence": 1.0,
                        "verification_method": "local_database",
                        "product": {
                            "name": product.name,
                            "manufacturer": product.manufacturer,
                            "product_code": product.product_code,
                            "manufacturing_date": product.manufacturing_date.isoformat() if product.manufacturing_date else None,
                            "batch_number": product.batch_number
                        }
                    }
                    return self._process_verification_result(result)
            except Exception as e:
                # Continue with online verification if local lookup fails
                pass

            # Search online databases
            online_results = await self.search_online(barcode)
            
            if not online_results:
                return self._process_verification_result({
                    "status": "error",
                    "message": "No product information found",
                    "confidence": 0.0,
                    "verification_method": "online_lookup"
                })
            
            # Combine results and calculate confidence
            combined_info = self._combine_results(online_results)
            confidence = self._calculate_confidence(online_results)
            
            # Determine verification status
            if confidence >= 0.8:
                status = "success"
                message = "Product verified successfully"
            elif confidence >= 0.5:
                status = "warning"
                message = "Product verification inconclusive"
            else:
                status = "error"
                message = "Product verification failed"
            
            result = {
                "status": status,
                "message": message,
                "confidence": confidence,
                "verification_method": "online_lookup",
                "product_info": combined_info,
                "sources_checked": len(online_results)
            }
            
            return self._process_verification_result(result)

        except Exception as e:
            return self._process_verification_result({
                "status": "error",
                "message": str(e),
                "confidence": 0.0,
                "verification_method": "barcode_verification"
            }) 