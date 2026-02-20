"""
Core module for curve digitization functionality.

Includes OpenAI API client and image processing capabilities.
"""

from .openai_client import OpenAIClient
from .image_processor import CurveDigitizer

__all__ = ['OpenAIClient', 'CurveDigitizer']
