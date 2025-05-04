from textwrap import dedent
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProductVerificationAgent:
    def __init__(self):
        """Initialize the product verification agent"""
        self.agent = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=os.getenv('GOOGLE_API_KEY')),
            description=dedent("""\
                You are an expert product verification specialist with extensive knowledge of authentic product packaging,
                branding, and design elements. Your expertise helps identify counterfeit products by analyzing visual
                characteristics and comparing them with authentic products.\
            """),
            instructions=dedent("""\
                When analyzing product images, follow these verification principles:

                1. Packaging Analysis:
                   - Examine the material and type of packaging (plastic, glass, etc.)
                   - Check for correct dimensions and proportions
                   - Verify packaging quality and finish
                   - Look for tamper-evident features

                2. Branding Verification:
                   - Compare logos with authentic versions
                   - Verify font styles and sizes
                   - Check color accuracy and consistency
                   - Examine label placement and alignment

                3. Product Details:
                   - Verify product information accuracy
                   - Check barcodes and serial numbers
                   - Examine manufacturing details
                   - Look for spelling errors or inconsistencies

                4. Counterfeit Indicators:
                   - Identify common counterfeit markers
                   - Note any suspicious elements
                   - Compare with known authentic products
                   - Consider regional variations

                Provide a detailed analysis with confidence levels and specific reasons for your conclusions.\
            """),
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
            markdown=True,
        )

    def verify_product(self, image_data: bytes, product_name: str) -> dict:
        """
        Analyze a product image and provide verification results.
        
        Args:
            image_data: Raw image data in bytes
            product_name: Name of the product to verify
            
        Returns:
            dict: Verification results including analysis and confidence level
        """
        try:
            # Create Image object from bytes
            image = Image(content=image_data)
            
            # Prepare the verification prompt
            prompt = f"""
            Analyze this {product_name} product image and determine if it appears to be authentic or counterfeit.
            Consider the following aspects:
            1. Packaging material and type
            2. Branding elements (logo, font, colors)
            3. Product information and labels
            4. Any suspicious or inconsistent elements
            
            Provide a detailed analysis with specific observations and a confidence level.
            """
            
            # Get the agent's response
            response = self.agent.run(
                prompt,
                images=[image],
            )
            
            # Extract confidence level and details
            confidence = self._extract_confidence(response.content)
            details = self._extract_details(response.content)
            
            # Determine overall status
            status = self._determine_status(confidence, details)
            
            return {
                "status": status,
                "message": response.content,
                "confidence": confidence,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error in product verification: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during verification: {str(e)}",
                "confidence": 0.0,
                "details": {}
            }

    def _extract_confidence(self, content: str) -> float:
        """Extract confidence level from the agent's response."""
        # This is a simple implementation - you might want to make it more sophisticated
        if "high confidence" in content.lower():
            return 0.9
        elif "medium confidence" in content.lower():
            return 0.6
        elif "low confidence" in content.lower():
            return 0.3
        return 0.5  # Default confidence if not specified

    def _extract_details(self, content: str) -> dict:
        """Extract key details from the agent's response."""
        details = {
            "packaging_issues": [],
            "branding_issues": [],
            "product_info_issues": [],
            "other_issues": []
        }
        
        # Simple keyword-based extraction
        if "packaging" in content.lower():
            details["packaging_issues"].append("Packaging analysis available")
        if "logo" in content.lower() or "brand" in content.lower():
            details["branding_issues"].append("Branding analysis available")
        if "product information" in content.lower():
            details["product_info_issues"].append("Product information analysis available")
            
        return details

    def _determine_status(self, confidence: float, details: dict) -> str:
        """Determine the overall verification status based on confidence and details."""
        if confidence >= 0.8:
            return "success"
        elif confidence >= 0.5:
            return "warning"
        else:
            return "error" 