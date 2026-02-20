"""
Google Gemini API Integration Module

Provides utilities for interacting with Google Gemini API for:
- Intent classification
- Image analysis and feature extraction
- Structured data field classification
"""

import google.generativeai as genai
import base64
import json
from typing import Dict, Any, Optional


class GeminiClient:
    """Client for Google Gemini API interactions."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google Gemini API key
            model: Model name to use (default: gemini-1.5-flash)
        """
        genai.configure(api_key=api_key)
        self.model = model
        self.client = genai.GenerativeModel(model)
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user query as 'imageprocessing' or 'nonimageprocessing'.
        
        Args:
            query: User's input query
            
        Returns:
            Classification result: 'imageprocessing' or 'nonimageprocessing'
        """
        prompt = f"""Classify this query as either 'imageprocessing' or 'nonimageprocessing'.

Query: "{query}"

Rules:
- If the query involves extracting data from images, analyzing charts/curves, or image manipulation, classify as 'imageprocessing'
- If the query involves analyzing structured data (JSON, CSV, text records), classify as 'nonimageprocessing'

Respond with ONLY one word: 'imageprocessing' or 'nonimageprocessing'"""
        
        response = self.client.generate_content(prompt)
        classification = response.text.strip().lower()
        
        # Validate response
        if classification not in ['imageprocessing', 'nonimageprocessing']:
            # Try to extract from response
            if 'imageprocessing' in classification or 'image' in classification:
                classification = 'imageprocessing'
            else:
                classification = 'nonimageprocessing'
        
        return classification
    
    def extract_axis_info(self, image_base64: str, query: str = "") -> Dict[str, Any]:
        """
        Extract axis information from a base64-encoded performance curve image.
        
        Args:
            image_base64: Base64-encoded image string
            query: Optional context about the image
            
        Returns:
            Dictionary with xMin, xMax, yMin, yMax, xUnit, yUnit
        """
        prompt = f"""Analyze this base64-encoded image of a performance curve and extract the following:

{f"Context: {query}" if query else ""}

Please extract and return as JSON:
{{
  "xMin": <minimum x-axis value>,
  "xMax": <maximum x-axis value>,
  "yMin": <minimum y-axis value>,
  "yMax": <maximum y-axis value>,
  "xUnit": "<unit of measurement for x-axis>",
  "yUnit": "<unit of measurement for y-axis>",
  "imageDescription": "<brief description of the chart>"
}}

If axis values cannot be determined, use null. Return ONLY valid JSON."""
        
        # Create image data for Gemini
        image_data = {
            "mime_type": "image/png",
            "data": image_base64
        }
        
        response = self.client.generate_content([prompt, image_data])
        
        # Parse JSON response
        try:
            result_text = response.text.strip()
            # Extract JSON from response if wrapped in markdown
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            axis_info = json.loads(result_text)
            return axis_info
        except json.JSONDecodeError as e:
            print(f"Error parsing axis info JSON: {e}")
            print(f"Response: {response.text}")
            return {
                "xMin": None, "xMax": None, "yMin": None, "yMax": None,
                "xUnit": "unknown", "yUnit": "unknown",
                "imageDescription": response.text
            }
    
    def extract_curve_features(self, image_base64: str) -> Dict[str, Any]:
        """
        Extract descriptions of curves and lines from performance curve image.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            Dictionary with list of features/curves detected
        """
        prompt = """Analyze this performance curve image and describe all curves and lines visible.

For each curve/line, provide:
- Color (specific color name like 'red', 'blue', 'green', etc.)
- Shape (curved, straight, stepped, etc.)
- Label or description if visible
- General position/trend (increasing, decreasing, constant, etc.)

Return as JSON:
{
  "curves": [
    {
      "color": "<color name>",
      "shape": "<shape description>",
      "label": "<label or description>",
      "trend": "<increasing|decreasing|constant|variable>",
      "approximate_values": "<range of values the curve spans>"
    }
  ],
  "numerical_data_visible": true/false,
  "grid_present": true/false,
  "overall_description": "<brief high-level description>"
}

Return ONLY valid JSON."""
        
        image_data = {
            "mime_type": "image/png",
            "data": image_base64
        }
        
        response = self.client.generate_content([prompt, image_data])
        
        try:
            result_text = response.text.strip()
            # Extract JSON from response if wrapped in markdown
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            features = json.loads(result_text)
            return features
        except json.JSONDecodeError as e:
            print(f"Error parsing curve features JSON: {e}")
            print(f"Response: {response.text}")
            return {"curves": [], "error": response.text}
    
    def get_general_response(self, query: str) -> str:
        """
        Get a general conversational response from Gemini.
        
        Args:
            query: User's query or message
            
        Returns:
            Gemini's response text
        """
        prompt = f"""You are a helpful assistant for performance curve digitization and engineering analysis.
        
The user has asked: {query}

Provide a helpful, conversational response. Be concise but informative."""
        
        response = self.client.generate_content(prompt)
        return response.text.strip()
    
    def classify_data_field(self, field_name: str, field_value: str) -> Dict[str, str]:
        """
        Classify a structured data field value.
        
        Args:
            field_name: Name of the field
            field_value: Value of the field
            
        Returns:
            Dictionary with classification and confidence
        """
        prompt = f"""Classify the following data field:

Field Name: {field_name}
Field Value: {field_value}

Classify as one of:
- AGE: numeric age value (e.g., 25, 45)
- IP_ADDRESS: IPv4 or IPv6 address
- EMAIL: email address format
- LOCATION: city, country, or geographic location
- TOKEN: authentication token, API key, or similar
- PHONE: phone number
- NAME: person's name
- CREDIT_CARD: credit card number pattern
- URL: web address
- IDENTIFIER: ID number, UUID, or similar
- NONE: generic or unclear data

Respond with ONLY a JSON object:
{{"classification": "<CLASS>", "confidence": <0.0-1.0>, "reason": "<brief explanation>"}}"""
        
        response = self.client.generate_content(prompt)
        
        try:
            result_text = response.text.strip()
            # Extract JSON if wrapped
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            classification = json.loads(result_text)
            return classification
        except json.JSONDecodeError as e:
            print(f"Error parsing classification JSON: {e}")
            return {
                "classification": "NONE",
                "confidence": 0.0,
                "reason": f"Failed to parse: {response.text}"
            }
