from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseVerificationAgent(ABC):
    """Base class for verification agents."""

    def __init__(self):
        self.verification_type = "base"
        self.metadata = {
            "verification_type": self.verification_type,
            "version": "2.0.0",
        }

    @abstractmethod
    async def verify_authenticity(self, **kwargs) -> Dict[str, Any]:
        """Verify the authenticity of a product. Returns a dict of results."""
        raise NotImplementedError
