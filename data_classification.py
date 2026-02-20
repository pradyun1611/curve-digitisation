"""
Data Classification Module

Handles structured data analysis and field classification:
- Loads JSON/CSV data
- Classifies each field using Gemini
- Outputs labeled classification results
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from gemini_api import GeminiClient


class DataClassifier:
    """Classifies structured data fields."""
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize data classifier.
        
        Args:
            gemini_client: Instance of GeminiClient
        """
        self.gemini_client = gemini_client
    
    def load_json_data(self, file_path: str) -> Union[Dict, List]:
        """
        Load JSON data from file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_csv_data(self, file_path: str) -> List[Dict]:
        """
        Load CSV data from file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of dictionaries representing rows
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def classify_json_structure(self, data: Union[Dict, List], max_depth: int = 3) -> Dict[str, Any]:
        """
        Classify all fields in a JSON structure.
        
        Args:
            data: JSON data (dict or list)
            max_depth: Maximum nesting depth to process
            
        Returns:
            Classified structure with sensitivity labels
        """
        results = {
            'data_type': 'json',
            'classifications': [],
            'summary': {}
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                classification = self._classify_field(key, value)
                results['classifications'].append({
                    'field_name': key,
                    'field_value': str(value)[:100],  # Truncate for display
                    'classification': classification
                })
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Classify fields from first record
                for key, value in data[0].items():
                    classification = self._classify_field(key, value)
                    results['classifications'].append({
                        'field_name': key,
                        'field_value': str(value)[:100],
                        'classification': classification,
                        'applies_to': f'all {len(data)} records'
                    })
        
        # Generate summary
        results['summary'] = self._generate_classification_summary(results['classifications'])
        
        return results
    
    def classify_csv_structure(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Classify all columns in CSV data.
        
        Args:
            data: List of row dictionaries
            
        Returns:
            Classified structure with sensitivity labels
        """
        results = {
            'data_type': 'csv',
            'total_rows': len(data),
            'columns': [],
            'summary': {}
        }
        
        if not data:
            return results
        
        # Analyze each column
        first_row = data[0]
        for key in first_row.keys():
            # Use first row value for classification
            value = first_row[key]
            classification = self._classify_field(key, value)
            
            # Collect sample values
            samples = [row[key] for row in data[:min(3, len(data))]]
            
            results['columns'].append({
                'column_name': key,
                'classification': classification,
                'sample_values': samples,
                'value_count': len(set(row[key] for row in data))
            })
        
        # Generate summary
        results['summary'] = self._generate_classification_summary(results['columns'])
        
        return results
    
    def _classify_field(self, field_name: str, field_value: Any) -> Dict[str, Any]:
        """
        Use Gemini to classify a single field.
        
        Args:
            field_name: Field name
            field_value: Field value
            
        Returns:
            Classification dictionary
        """
        classification_result = self.gemini_client.classify_data_field(
            field_name, str(field_value)
        )
        
        return {
            'classification': classification_result.get('classification', 'NONE'),
            'confidence': classification_result.get('confidence', 0.0),
            'reason': classification_result.get('reason', '')
        }
    
    def _generate_classification_summary(self, classifications: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics of classifications."""
        summary = {
            'total_fields': len(classifications),
            'classification_counts': {},
            'sensitive_fields': [],
            'high_confidence_count': 0
        }
        
        sensitive_classes = {'IP_ADDRESS', 'EMAIL', 'TOKEN', 'PHONE', 'CREDIT_CARD'}
        
        for item in classifications:
            class_type = item['classification'].get('classification', 'NONE')
            confidence = item['classification'].get('confidence', 0.0)
            
            # Count classifications
            summary['classification_counts'][class_type] = \
                summary['classification_counts'].get(class_type, 0) + 1
            
            # Track sensitive fields
            if class_type in sensitive_classes:
                summary['sensitive_fields'].append({
                    'field_name': item.get('field_name', item.get('column_name', 'unknown')),
                    'classification': class_type,
                    'confidence': confidence
                })
            
            # Track high confidence
            if confidence >= 0.8:
                summary['high_confidence_count'] += 1
        
        summary['confidence_average'] = \
            sum(c['classification'].get('confidence', 0.0) for c in classifications) / len(classifications) \
            if classifications else 0.0
        
        return summary
    
    def save_classification_report(self, classification: Dict[str, Any], 
                                  output_path: str) -> None:
        """
        Save classification results to JSON file.
        
        Args:
            classification: Classification results dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(classification, f, indent=2)
    
    def generate_redaction_mask(self, data: Union[Dict, List], 
                               classification: Dict[str, Any]) -> Union[Dict, List]:
        """
        Generate a mask indicating which fields should be redacted.
        
        Args:
            data: Original data
            classification: Classification results
            
        Returns:
            Mask indicating sensitive fields
        """
        sensitive_classes = {'IP_ADDRESS', 'EMAIL', 'TOKEN', 'PHONE', 'CREDIT_CARD', 'LOCATION'}
        
        mask = {}
        
        if 'classifications' in classification:
            for item in classification['classifications']:
                field_name = item.get('field_name', '')
                class_type = item['classification'].get('classification', 'NONE')
                confidence = item['classification'].get('confidence', 0.0)
                
                should_redact = class_type in sensitive_classes and confidence >= 0.7
                mask[field_name] = {
                    'redact': should_redact,
                    'classification': class_type,
                    'confidence': confidence
                }
        
        elif 'columns' in classification:
            for item in classification['columns']:
                col_name = item.get('column_name', '')
                class_type = item['classification'].get('classification', 'NONE')
                confidence = item['classification'].get('confidence', 0.0)
                
                should_redact = class_type in sensitive_classes and confidence >= 0.7
                mask[col_name] = {
                    'redact': should_redact,
                    'classification': class_type,
                    'confidence': confidence
                }
        
        return mask
