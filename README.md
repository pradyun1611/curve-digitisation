# Performance Curve Digitization Chatbot

An interactive Python chatbot that digitizes performance curve images using Azure OpenAI's GPT models. The chatbot classifies user queries to determine if they require image processing or general responses.

## Features

### Chatbot Capabilities
- **Conversational Interface**: Interactive chat with user-friendly prompts
- **Intent Classification**: Automatically detect if query requires image processing or general response
- **Image Processing Flow** (when intent = "imageprocessing"):
  - Extract axis boundaries and units from charts using OpenAI's vision
  - Identify curves, colors, and labels in images
  - Extract pixel coordinates for specific colors
  - Normalize pixels to axis coordinates
  - Remove noise using RANSAC filtering
  - Fit quadratic polynomial curves with R² metrics
  - Export results to JSON with coefficients and points
- **General Responses**: Answer questions conversationally when not image-related
- **Auto Image Detection**: Automatically detect image file paths in messages

## Project Structure

```
curve-digitisation/
├── core/                         # Core functionality modules
│   ├── __init__.py              # Package initialization
│   ├── openai_client.py         # OpenAI API client
│   └── image_processor.py       # Image digitization and curve fitting
│
├── ui/                          # User interface modules (Streamlit)
│   ├── __init__.py              # Package initialization
│   ├── sidebar.py               # Sidebar configuration
│   ├── chat_interface.py        # Chat message display and history
│   └── result_display.py        # Result visualization and metrics
│
├── input/                       # Input directory for images
│   └── .gitkeep
│
├── output/                      # Output directory for results
│   └── .gitkeep
│
├── requirements.txt             # Project dependencies
├── main.py                      # CLI chatbot entry point
├── streamlit_app.py             # Streamlit web UI entry point
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

### Module Descriptions

**core/openai_client.py**
- `OpenAIClient`: OpenAI API wrapper class
- Methods: classify_intent, extract_axis_info, extract_curve_features, get_general_response

**core/image_processor.py**
- `CurveDigitizer`: Image processing and curve fitting class
- Methods: load_image, extract_color_pixels, normalize_to_axis, clean_coordinates_ransac, fit_polynomial_curve

**ui/sidebar.py**
- `setup_sidebar()`: Configures sidebar with API key input and settings

**ui/chat_interface.py**
- `display_chat_message()`: Display single message
- `display_chat_history()`: Display all messages
- `initialize_session_state()`: Initialize Streamlit state

**ui/result_display.py**
- `display_image_results()`: Visualize processing results
- `display_processing_summary()`: Generate result summary text

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- Azure OpenAI deployment with access to an endpoint and API key

### 2. Clone/Download Project

```bash
cd curve-digitisation
```

### 3. Setup Virtual Environment (Recommended)

On Windows PowerShell:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

On Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Azure OpenAI Credentials

**Option A: Environment Variables (recommended)**

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY = 'your-azure-api-key'
$env:AZURE_ENDPOINT = 'https://your-resource.openai.azure.com/'
$env:AZURE_DEPLOYMENT_NAME = 'gpt-5-chat'
```

Linux/macOS:
```bash
export OPENAI_API_KEY='your-azure-api-key'
export AZURE_ENDPOINT='https://your-resource.openai.azure.com/'
export AZURE_DEPLOYMENT_NAME='gpt-5-chat'
```

**Option B: Input in UI**

When running the Streamlit app, enter your API key in the sidebar text field.

### 5. Run the Application

**Start Streamlit Web UI:**
```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

**Or use CLI chatbot:**
```bash
python main.py
```

## Usage

### Option 1: Streamlit Web UI (Recommended)

The easiest way to use the chatbot is through the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

This opens an interactive web app in your browser with:
- Chat interface
- Image upload and processing
- Real-time result visualization
- JSON export functionality
- Settings sidebar for API key configuration

**Streamlit UI Features:**
- 📸 Drag-and-drop image upload
- 💬 Chat message history
- 📊 Interactive result display with expandable curve details
- 📥 Download results as JSON
- 🎨 Professional UI with custom styling
- ⚙️ Configurable settings sidebar

### Option 2: Command-Line Chatbot

For terminal-based interaction:

```bash
python main.py [--api-key YOUR_KEY] [--output ./results/]
```

Then interact naturally in the terminal:
```
You: Extract curves from this chart performance.png
Bot: ✓ Image analysis complete!
     Curves detected: 2...

You: What is polynomial fitting?
Bot: Polynomial fitting is a method...

You: quit
Bot: Goodbye! 👋
```

## Streamlit Web UI Guide

### Interface Overview

**Left Sidebar:**
- 🔑 API Key input field
- 📁 Output directory configuration
- 📚 Usage instructions and feature overview
- 🗑️ Clear history button

**Main Chat Area:**
- 💬 Conversation history display
- 📝 Message input field
- 📸 Image upload button
- 📊 Real-time result visualization

### Uploading and Processing Images

1. Enter your query in the text field (e.g., "Extract curves from this chart")
2. Upload an image using the "📸 Upload an image" button
3. Click "Send" to process
4. View results in the expandable sections below

### Viewing Results

Results are displayed in organized sections:
- **Axis Information**: X and Y axis ranges and units
- **Image Dimensions**: Width and height in pixels
- **Detected Curves**: Expandable cards for each curve showing:
  - Original point count
  - Cleaned point count (after RANSAC filtering)
  - R² fit score
  - Polynomial equation and coefficients
  - Error messages (if any)

### Downloading Results

Each image processing session generates a JSON file. Download it using the "📥 Download Results (JSON)" button to get:
- Full curve data
- Axis information
- Polynomial coefficients
- All extracted points
- Processing metadata

## Output Files

### Image Processing Results

**File**: `curve_digitization_YYYYMMDD_HHMMSS.json`

Example output structure:
```json
{
  "image_path": "performance_curve.png",
  "image_dimensions": {
    "width": 800,
    "height": 600
  },
  "axis_info": {
    "xMin": 0,
    "xMax": 100,
    "yMin": 0,
    "yMax": 120,
    "xUnit": "Speed (%)",
    "yUnit": "Efficiency (%)"
  },
  "curves": {
    "red": {
      "label": "Model A",
      "color": "red",
      "original_point_count": 2541,
      "normalized_point_count": 2541,
      "cleaned_point_count": 2314,
      "fit_result": {
        "degree": 2,
        "coefficients": [0.0523, -5.234, 98.45],
        "r_squared": 0.9876,
        "equation": "0.0523*x^2 - 5.234*x + 98.45",
        "fitted_points": [
          {"x": 0.0, "y": 98.45},
          {"x": 2.04, "y": 88.23},
          ...
        ]
      }
    }
  }
}
```

## Module Documentation

### openai_client.py

**OpenAIClient** class handles all Azure OpenAI API interactions.

#### Methods

- `classify_intent(query: str) -> str`
  - Classifies query as 'imageprocessing' or 'nonimageprocessing'
  - Determines if user wants image analysis or general response

- `extract_axis_info(image_base64: str, query: str) -> Dict`
  - Extracts axis boundaries and units from chart image via Azure OpenAI vision
  - Returns: `{xMin, xMax, yMin, yMax, xUnit, yUnit, imageDescription}`

- `extract_curve_features(image_base64: str) -> Dict`
  - Describes curves, colors, labels, and trends in image using Azure OpenAI vision
  - Returns: `{curves: [{color, shape, label, trend}], numerical_data_visible, grid_present}`

- `get_general_response(query: str) -> str`
  - Get conversational response for non-image queries
  - Returns: Plain text response from OpenAI GPT

### image_processing.py

**CurveDigitizer** class handles image analysis and curve extraction.

#### Methods

- `load_image(image_path: str) -> Image`
  - Load PNG/JPG image from file

- `crop_image(image: Image, crop_box: tuple) -> Image`
  - Crop image to specified region (left, top, right, bottom)

- `extract_color_pixels(image: Image, target_color_name: str) -> List[Tuple]`
  - Extract pixel coordinates matching a specific color
  - Supports: red, blue, green, yellow, orange, purple, black, gray

- `normalize_to_axis(pixel_coords: List, width: int, height: int) -> List`
  - Convert pixel coordinates to axis coordinates

- `clean_coordinates_ransac(coordinates: List, threshold: float) -> List`
  - Remove outliers using RANSAC-like filtering (threshold: 0.1 = 10% of range)

- `fit_polynomial_curve(coordinates: List, degree: int) -> Dict`
  - Fit polynomial (default degree 2) to coordinates
  - Returns: `{degree, coefficients, r_squared, fitted_points, equation}`

- `process_curve_image(image_path: str, features: List) -> Dict`
  - End-to-end: extract → normalize → clean → fit

## Chatbot Query Classification

The chatbot automatically classifies queries using this logic:

**Image Processing** (↓ processes image):
- "Extract curves from..."
- "Digitize this chart"
- "Analyze this graph"
- "Get data from this image"
- "Process this performance curve"

**General Response** (→ responds conversationally):
- "What is..."
- "How do I..."
- "Explain..."
- "Tell me about..."
- Any query without image processing keywords

## OpenAI Prompts Used

### Intent Classification
```
Classify this query as either 'imageprocessing' or 'nonimageprocessing'.

Query: "{user_query}"

Rules:
- If the query involves extracting data from images, analyzing charts/curves, or image manipulation, 
  classify as 'imageprocessing'
- If the query involves analyzing structured data or general questions, classify as 'nonimageprocessing'

Respond with ONLY one word: 'imageprocessing' or 'nonimageprocessing'
```

### Axis Extraction
```
Analyze this image of a performance curve and extract axis information:
- Min/max values for both axes
- Units of measurement
- Chart description

Return as JSON: {xMin, xMax, yMin, yMax, xUnit, yUnit, imageDescription}
```

### Curve Feature Detection
```
Describe all curves and lines visible in this performance curve image:
- Color (e.g., 'red', 'blue')
- Shape (curved, straight, stepped)
- Label or description
- Trend (increasing, decreasing, constant)

Return as JSON with curves array.
```

## Processing Pipeline

1. **User Input** - Accept query (with optional image file path)
2. **Intent Classification** - Azure OpenAI determines if imageprocessing or general
3. **Image Processing Path** (if imageprocessing):
   - Load and encode image to base64
   - Extract axis information from image using Azure OpenAI vision
   - Identify curve features and colors using Azure OpenAI vision
   - Extract color-specific pixels
   - Normalize pixel coordinates to axis values
   - Apply RANSAC filtering to remove noise
   - Fit quadratic polynomial to each curve
   - Save results to JSON
4. **General Response Path** (if general):
   - Send query to Azure OpenAI GPT
   - Return conversational response

## Error Handling

- Missing API key: Clear error message with setup instructions
- Missing image file: Prompts user to provide image path
- Invalid image format: Graceful error with suggestion
- JSON parsing errors: Returns raw response with error note
- RANSAC failures: Falls back to original coordinates

## Performance Tips

- **Large Images**: Crop the region of interest for faster processing
- **API Rate Limits**: Space out requests for production use
- **Noisy Data**: Increase RANSAC threshold if many points are filtered
- **Polynomial Fit**: Default uses degree 2 (quadratic); modify in code if needed

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"

Set the environment variable before running:
```powershell
$env:OPENAI_API_KEY = 'your-azure-key'
python main.py
```

### "AZURE_ENDPOINT environment variable not set"

Set the Azure endpoint URL:
```powershell
$env:AZURE_ENDPOINT = 'https://your-resource.openai.azure.com/'
python main.py
```

### "AZURE_DEPLOYMENT_NAME environment variable not set"

Set the deployment name from your Azure OpenAI resource:
```powershell
$env:AZURE_DEPLOYMENT_NAME = 'gpt-5-chat'
python main.py
```

### "Module not found: openai"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Image detection not working

Make sure image file path is included in your message:
```
You: "Extract curves from image.png"  # ✓ Will work
You: "Extract curves"                  # ✗ Won't detect image
```

### Poor curve fitting results

Causes and solutions:
| Problem | Cause | Solution |
|---------|-------|----------|
| R² < 0.8 | Noisy/scattered data | Increase RANSAC threshold to 0.15 |
| R² < 0.8 | Non-polynomial curve | Try degree 3 (cubic) instead of 2 |
| Wrong color extraction | Color range mismatch | Update color_ranges in extract_color_pixels() |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| openai | >=1.0.0 | OpenAI API client |
| Pillow | >=11.0.0 | Image processing |
| numpy | >=1.24.3 | Array operations |
| scikit-learn | >=1.3.2 | RANSAC, polynomial fitting |
| requests | >=2.31.0 | HTTP requests |
| streamlit | >=1.28.0 | Web UI framework |

## Future Enhancements

- [ ] Multi-image processing in single conversation
- [ ] Batch processing mode for multiple images
- [ ] Support for PDF charts and SVG images
- [ ] Interactive curve annotation UI in Streamlit
- [ ] OCR for automatic axis label detection
- [ ] Spline fitting (B-splines, cubic splines)
- [ ] Export to CSV/Excel format
- [ ] Dark mode for Streamlit UI
- [ ] Async processing for faster image handling
- [ ] Model comparison (Gemini vs local vision models)

## License

MIT License - Use freely for commercial and personal projects.

---

**Built with**: Python, Azure OpenAI GPT, Streamlit, scikit-learn, Pillow
