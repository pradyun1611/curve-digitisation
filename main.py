"""
Chatbot Interface for Performance Curve Digitization

Interactive chatbot that:
- Accepts user queries and optional image files
- Classifies intent as "imageprocessing" or "nonimageprocessing"
- Processes images (extract curves) or responds generally with Azure OpenAI
- Provides conversational responses

Usage:
    python main.py [--api-key YOUR_KEY] [--endpoint URL] [--deployment NAME] [--output ./output/]

Example:
    python main.py
    > User: Show me the curves in this performance chart
    > [User provides image.png]
    > Bot: [Processes image and extracts curves]
"""

import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.openai_client import OpenAIClient
from core.image_processor import CurveDigitizer


def load_image_as_base64(image_path: str) -> str:
    """
    Load image file and encode as base64.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64-encoded image string
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def process_image_task(openai_client: OpenAIClient, image_path: str, user_query: str,
                       output_dir: str) -> str:
    """
    Execute image processing flow and return summary response.
    
    Args:
        openai_client: Initialized OpenAI client
        image_path: Path to image file
        user_query: User's query about the image
        output_dir: Directory to save results
        
    Returns:
        Response message to user
    """
    try:
        # Encode image to base64
        image_base64 = load_image_as_base64(image_path)
        
        # Extract axis information
        axis_info = openai_client.extract_axis_info(image_base64, user_query)
        
        # Extract curve features
        features = openai_client.extract_curve_features(image_base64)
        curve_count = len(features.get('curves', []))
        
        # Digitize curves and generate graphs
        digitizer = CurveDigitizer(axis_info)
        results = digitizer.process_curve_image(image_path, features, output_dir)
        
        # Save JSON results in the per-instance output folder
        instance_dir = Path(results.get('instance_dir', output_dir))
        output_file = instance_dir / "curve_digitization.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Build response
        response = f"✓ Image analysis complete!\n\n"
        response += f"**Axis Information:**\n"
        response += f"- X-axis: {axis_info.get('xMin')} to {axis_info.get('xMax')} ({axis_info.get('xUnit')})\n"
        response += f"- Y-axis: {axis_info.get('yMin')} to {axis_info.get('yMax')} ({axis_info.get('yUnit')})\n\n"
        response += f"**Curves Detected:** {curve_count}\n"
        
        for curve in features.get('curves', []):
            color = curve.get('color', 'unknown')
            label = curve.get('label', 'unlabeled')
            response += f"- {color.capitalize()}: {label}\n"
        
        # Overall graph-level metrics
        overall = results.get('overall_metrics', {})
        if overall and overall.get('delta_value') is not None:
            response += f"\n**Overall Graph Quality** (averaged across {overall.get('curve_count', 0)} curves):\n"
            response += f"  - Delta Value: {overall['delta_value']:.4f}\n"
            response += f"  - Delta Norm:  {overall['delta_norm']:.4%}\n"
            response += f"  - Delta P95:   {overall['delta_p95']:.4f}\n"
            response += f"  - IoU:         {overall['iou']:.4f}\n"
            response += f"  - Precision:   {overall['precision']:.4f}\n"
            response += f"  - Recall:      {overall['recall']:.4f}\n"
        
        # Quality metrics per curve
        curves_data = results.get('curves', {})
        has_metrics = False
        for cname, cdata in curves_data.items():
            m = cdata.get('metrics', {})
            if m and m.get('delta_value') is not None:
                if not has_metrics:
                    response += f"\n**Quality Metrics:**\n"
                    has_metrics = True
                label = cdata.get('label', cname)
                response += f"\n  __{label}__\n"
                response += f"  - Delta Value: {m['delta_value']:.4f}  (mean abs error in axis units)\n"
                response += f"  - Delta Norm:  {m['delta_norm']:.4%}  (normalized by Y range)\n"
                response += f"  - Delta P95:   {m['delta_p95']:.4f}  (95th percentile error)\n"
                response += f"  - IoU:         {m['iou']:.4f}  (overlap: 1.0 = perfect)\n"
                response += f"  - Precision:   {m['precision']:.4f}  (fitted curve quality)\n"
                response += f"  - Recall:      {m['recall']:.4f}  (data capture rate)\n"
        
        response += f"\n**Results saved to:** {output_file}\n"
        
        # List generated graphs
        output_graphs = results.get('output_graphs', {})
        if output_graphs:
            response += f"\n**Digitized graphs saved:**\n"
            for name, path in output_graphs.items():
                response += f"- {name}: {path}\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error processing image: {str(e)}"


def get_general_response(openai_client: OpenAIClient, user_query: str) -> str:
    """
    Get a general conversational response from OpenAI GPT.
    
    Args:
        openai_client: Initialized OpenAI client
        user_query: User's query
        
    Returns:
        OpenAI's response
    """
    try:
        response = openai_client.get_general_response(user_query)
        return response
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def chatbot_main(api_key: str, endpoint: str, deployment_name: str, output_dir: str = "./output/"):
    """
    Main chatbot loop.
    
    Args:
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment_name: Azure deployment name
        output_dir: Directory to save results
    """
    print("="*70)
    print("🤖 Curve Digitization Chatbot")
    print("="*70)
    print("I can help you:")
    print("  • Extract and digitize curves from performance chart images")
    print("  • Answer general questions\n")
    print("To process an image, describe the task and provide the image path.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Initialize Azure OpenAI client
    openai_client = OpenAIClient(api_key, endpoint, deployment_name)
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nBot: Goodbye! 👋")
                break
            
            # Check for image file reference
            image_path = None
            query = user_input
            
            # Simple check for image file patterns
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                if ext in user_input.lower():
                    # Extract potential file path
                    words = user_input.split()
                    for word in words:
                        if ext in word.lower():
                            if Path(word).exists():
                                image_path = word
                                # Remove image path from query for cleaner processing
                                query = user_input.replace(word, '').strip()
                                break
                    break
            
            # Classify intent
            intent = openai_client.classify_intent(query)
            
            # Process based on intent
            if intent == 'imageprocessing':
                if image_path:
                    response = process_image_task(openai_client, image_path, query, output_dir)
                else:
                    response = "Bot: I detected this is about image processing, but I don't see an image file. " \
                              "Please provide the image path in your message (e.g., 'process image.png: extract curves')"
            else:  # nonimageprocessing
                response = get_general_response(openai_client, query)
            
            print(f"\nBot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nBot: ❌ An error occurred: {str(e)}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Curve Digitization Chatbot - Interactive tool for extracting curves from images"
    )
    parser.add_argument('--api-key', help='Azure OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--endpoint', help='Azure OpenAI endpoint URL (or set AZURE_ENDPOINT env var)')
    parser.add_argument('--deployment', help='Azure deployment name (or set AZURE_DEPLOYMENT_NAME env var)')
    parser.add_argument('--output', '-o', default='./output/',
                       help='Output directory for results (default: ./output/')
    
    args = parser.parse_args()
    
    # Get Azure credentials
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    endpoint = args.endpoint or os.getenv('AZURE_ENDPOINT')
    deployment_name = args.deployment or os.getenv('AZURE_DEPLOYMENT_NAME')
    
    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set it with:")
        print("  Windows: $env:OPENAI_API_KEY = 'your-api-key'")
        print("  Linux/Mac: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    if not endpoint:
        print("\nError: AZURE_ENDPOINT environment variable not set.")
        print("Please set it with:")
        print("  Windows: $env:AZURE_ENDPOINT = 'https://your-resource.openai.azure.com/'")
        print("  Linux/Mac: export AZURE_ENDPOINT='https://your-resource.openai.azure.com/'")
        sys.exit(1)
    
    if not deployment_name:
        print("\nError: AZURE_DEPLOYMENT_NAME environment variable not set.")
        print("Please set it with:")
        print("  Windows: $env:AZURE_DEPLOYMENT_NAME = 'gpt-5-chat'")
        print("  Linux/Mac: export AZURE_DEPLOYMENT_NAME='gpt-5-chat'")
        sys.exit(1)
    
    # Start chatbot
    chatbot_main(api_key, endpoint, deployment_name, args.output)
