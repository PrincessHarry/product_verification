from textwrap import dedent
from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductVerificationAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(api_key=os.getenv('GEMINI_API_KEY')),
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

    async def verify_product(self, image_path: str, product_name: str) -> dict:
        """
        Analyze a product image and provide verification results.
        
        Args:
            image_path: Path to the product image
            product_name: Name of the product to verify
            
        Returns:
            dict: Verification results including analysis and confidence level
        """
        try:
            # Create Image object from local file
            image = Image(path=image_path)
            
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
            response = await self.agent.run(
                prompt,
                images=[image],
            )
            
            return {
                "analysis": response.content,
                "confidence": self._extract_confidence(response.content),
                "details": self._extract_details(response.content)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "analysis": None,
                "confidence": None,
                "details": None
            }

    def _extract_confidence(self, content: str) -> str:
        """Extract confidence level from the agent's response."""
        # This is a simple implementation - you might want to make it more sophisticated
        if "high confidence" in content.lower():
            return "high"
        elif "medium confidence" in content.lower():
            return "medium"
        elif "low confidence" in content.lower():
            return "low"
        return "unknown"

    def _extract_details(self, content: str) -> dict:
        """Extract key details from the agent's response."""
        # This is a simple implementation - you might want to make it more sophisticated
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

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = ProductVerificationAgent()
        result = await agent.verify_product(
            image_path="path/to/product/image.jpg",
            product_name="Bama Mayonnaise"
        )
        print(result)
    
    asyncio.run(main()) 