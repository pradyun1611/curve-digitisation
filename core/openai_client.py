"""
Azure OpenAI API Integration Module

Provides utilities for interacting with Azure OpenAI API for:
- Intent classification
- Image analysis and feature extraction
- Structured data field classification
"""

from openai import AzureOpenAI
import base64
import json
from typing import Dict, Any, Optional


class OpenAIClient:
    """Client for Azure OpenAI API interactions using GPT models."""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, api_version: str = "2024-02-15-preview"):
        """
        Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)
            deployment_name: Deployment name in Azure (e.g., gpt-5-chat)
            api_version: API version (default: 2024-02-15-preview)
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    
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
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
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
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse JSON response
        try:
            result_text = response.choices[0].message.content.strip()
            # Extract JSON from response if wrapped in markdown
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            axis_info = json.loads(result_text)
            return axis_info
        except json.JSONDecodeError as e:
            print(f"Error parsing axis info JSON: {e}")
            print(f"Response: {result_text}")
            return {
                "xMin": None, "xMax": None, "yMin": None, "yMax": None,
                "xUnit": "unknown", "yUnit": "unknown",
                "imageDescription": result_text
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
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        try:
            result_text = response.choices[0].message.content.strip()
            # Extract JSON from response if wrapped in markdown
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            features = json.loads(result_text)
            return features
        except json.JSONDecodeError as e:
            print(f"Error parsing curve features JSON: {e}")
            print(f"Response: {result_text}")
            return {"curves": [], "error": result_text}
    
    def get_general_response(self, query: str) -> str:
        """
        Get a general conversational response from GPT.
        
        Args:
            query: User's query or message
            
        Returns:
            GPT's response text
        """
        prompt = f"""You are a helpful assistant for performance curve digitization and engineering analysis.
        
The user has asked: {query}

Provide a helpful, conversational response. Be concise but informative."""
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
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
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        try:
            result_text = response.choices[0].message.content.strip()
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
                "reason": f"Failed to parse: {response.choices[0].message.content}"
            }
