"""
Comprehensive data extraction utilities for BPO Automation Agent.
Handles parsing of MLS sheets, tax records, and other property-related documents.
"""

import os
import re
import logging
import pdfplumber
import pytesseract
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validation and normalization constants
VALID_PROPERTY_CLASSES = [
    'Residential', 'Commercial', 'Industrial', 'Agricultural', 'Vacant Land'
]

VALID_ZONING_TYPES = [
    'R-1', 'R-2', 'R-3', 'R-4', 'C-1', 'C-2', 'I-1', 'I-2', 'A-1'
]

VALID_SEWER_TYPES = ['Public', 'Private', 'Septic']
VALID_WATER_TYPES = ['Public', 'Private', 'Well']

# Compiled regex patterns for better performance
RE_NUMERIC = re.compile(r'[\d,]+')
RE_CURRENCY = re.compile(r'\$?([\d,]+)')
RE_DATE = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})')
RE_SQ_FT = re.compile(r'(?i)(\d+)\s*(?:sq\s*ft|square\s*feet)')
RE_YES_NO = re.compile(r'(?i)^(yes|no)$')

def validate_numeric(value: str, field_name: str) -> int:
    """
    Validate and convert a numeric string to an integer.
    
    Args:
        value: String value to validate
        field_name: Name of the field being validated
        
    Returns:
        Integer value
        
    Raises:
        ValueError: If value cannot be converted to integer
    """
    if not value:
        logger.warning(f"Empty value for {field_name}")
        return None
        
    try:
        # Remove any non-numeric characters except decimal point and minus sign
        cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
        if not cleaned:
            raise ValueError(f"Invalid numeric value for {field_name}: {value}")
        return int(float(cleaned))
    except ValueError as e:
        logger.warning(f"Invalid numeric value for {field_name}: {value}")
        raise ValueError(f"Invalid numeric value for {field_name}: {value}") from e

def validate_date(value: str, field_name: str) -> str:
    """
    Validate a date string in MM/DD/YYYY format.
    
    Args:
        value: Date string to validate
        field_name: Name of the field being validated
        
    Returns:
        Normalized date string
        
    Raises:
        ValueError: If value is not a valid date
    """
    if not value:
        logger.warning(f"Empty value for {field_name}")
        return None
        
    try:
        # Parse the date
        date_obj = datetime.strptime(value.strip(), '%m/%d/%Y')
        # Return in consistent format
        return date_obj.strftime('%m/%d/%Y')
    except ValueError as e:
        logger.warning(f"Invalid date value for {field_name}: {value}")
        raise ValueError(f"Invalid date value for {field_name}: {value}") from e

def validate_enum(value: str, valid_values, field_name: str) -> str:
    """
    Validate that a value is in the list or set of valid values.
    Returns the canonical value from valid_values.
    """
    if not value:
        logger.warning(f"Empty value for {field_name}")
        raise ValueError(f"Empty value for {field_name}")
    normalized = value.strip().lower()
    # Accept both set and list
    for valid in valid_values:
        if valid.lower() == normalized:
            return valid
    logger.warning(f"Invalid value for {field_name}: {value}")
    raise ValueError(f"Invalid value for {field_name}: {value}")

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and special characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
        
    # Convert to unicode and normalize
    text = unicodedata.normalize('NFKD', text)
    # Remove special characters
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()

def validate_yes_no(value: str, field_name: str) -> bool:
    """
    Validate a yes/no value.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        
    Returns:
        Boolean value
        
    Raises:
        ValueError: If value is not a valid yes/no value
    """
    if not value:
        logger.warning(f"Empty value for {field_name}")
        return None
        
    normalized = value.strip().lower()
    if normalized in ['yes', 'y', 'true', 't']:
        return True
    elif normalized in ['no', 'n', 'false', 'f']:
        return False
        
    logger.warning(f"Invalid yes/no value for {field_name}: {value}")
    raise ValueError(f"Invalid yes/no value for {field_name}: {value}")

def validate_sq_ft(value: str, field_name: str) -> int:
    """
    Validate a square footage value.
    
    Args:
        value: Value to validate
        field_name: Name of the field being validated
        
    Returns:
        Integer square footage value
        
    Raises:
        ValueError: If value is not a valid square footage
    """
    if not value:
        logger.warning(f"Empty value for {field_name}")
        return None
        
    try:
        # Remove any non-numeric characters except decimal point
        cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
        if not cleaned:
            raise ValueError(f"Invalid square footage value for {field_name}: {value}")
        return int(float(cleaned))
    except ValueError as e:
        logger.warning(f"Invalid square footage value for {field_name}: {value}")
        raise ValueError(f"Invalid square footage value for {field_name}: {value}") from e

class DocumentType(Enum):
    """Types of documents that can be parsed."""
    MLS_SHEET = "mls_sheet"
    TAX_RECORD = "tax_record"
    SUBJECT_PRIOR = "subject_prior"

@dataclass
class ParsingResult:
    """Container for parsing results with metadata."""
    data: Dict[str, Any]
    document_type: DocumentType
    warnings: List[str]
    errors: List[str]
    is_scanned: bool
    is_protected: bool

class ParsingError(Exception):
    """Base exception for parsing errors."""
    pass

class DocumentProtectedError(ParsingError):
    """Raised when a document is password protected."""
    pass

class ScannedDocumentError(ParsingError):
    """Raised when a document is scanned and requires OCR."""
    pass

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = os.getenv('BPO_CONFIG_PATH', 'bpo_agent_config')
    config_file = Path(config_path) / 'bpo_rules_and_settings.yaml'
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def is_document_protected(pdf_path: str) -> bool:
    """Check if a PDF is password protected."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Try to access the first page
            pdf.pages[0]
            return False
    except Exception as e:
        if "encrypted" in str(e).lower():
            logger.error(f"Password protected document detected: {pdf_path}")
            return True
        raise

def is_scanned_document(pdf_path: str) -> bool:
    """Check if a PDF appears to be scanned (image-based)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            # Check if the page has extractable text
            text = first_page.extract_text()
            if not text or len(text.strip()) < 50:  # Arbitrary threshold
                logger.warning(f"Document appears to be scanned: {pdf_path}")
                return True
            return False
    except Exception as e:
        logger.error(f"Error checking if document is scanned: {e}")
        return False

def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from a scanned PDF using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the document
        
    Raises:
        ParsingError: If OCR fails
    """
    logger.info(f"Performing OCR on document: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            
            for page in pdf.pages:
                # Convert page to image
                img = page.to_image()
                
                # Perform OCR
                text = pytesseract.image_to_string(img.original)
                all_text.append(text)
            
            return '\n'.join(all_text)
            
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise ParsingError(f"Failed to perform OCR: {e}")

def extract_mls_overview(pdf_path: str) -> Optional[str]:
    """
    Extract the Overview/Remarks section from an MLS sheet.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted overview text or None if not found
        
    Raises:
        ParsingError: If document cannot be parsed
        DocumentProtectedError: If document is password protected
        ScannedDocumentError: If document is scanned and requires OCR
    """
    logger.info(f"Extracting MLS overview from: {pdf_path}")
    
    if is_document_protected(pdf_path):
        raise DocumentProtectedError(f"Document is password protected: {pdf_path}")
    
    if is_scanned_document(pdf_path):
        raise ScannedDocumentError(f"Document appears to be scanned: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            overview_text = []
            in_overview = False
            
            # Common section headers that might indicate the start of overview
            overview_headers = [
                r"(?i)overview",
                r"(?i)public remarks",
                r"(?i)mls comments",
                r"(?i)property description",
                r"(?i)remarks"
            ]
            
            # Common section headers that might indicate the end of overview
            end_headers = [
                r"(?i)features",
                r"(?i)room information",
                r"(?i)property details",
                r"(?i)location",
                r"(?i)schools",
                r"(?i)disclaimer"
            ]
            
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                for line in lines:
                    # Check for start of overview section
                    if not in_overview:
                        if any(re.search(pattern, line) for pattern in overview_headers):
                            in_overview = True
                            continue
                    
                    # Check for end of overview section
                    if in_overview:
                        if any(re.search(pattern, line) for pattern in end_headers):
                            break
                        overview_text.append(line)
            
            if not overview_text:
                logger.warning(f"No overview section found in: {pdf_path}")
                return None
            
            # Clean and normalize the text
            cleaned_text = '\n'.join(overview_text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Normalize multiple newlines
            cleaned_text = re.sub(r' +', ' ', cleaned_text)  # Normalize spaces
            cleaned_text = cleaned_text.strip()
            
            logger.info(f"Successfully extracted overview from: {pdf_path}")
            return cleaned_text
            
    except Exception as e:
        logger.error(f"Error extracting MLS overview: {e}")
        raise ParsingError(f"Failed to extract MLS overview: {e}")

def extract_mls_sheet_data(pdf_path: str) -> Dict[str, Any]:
    """
    Extract all required data from an MLS sheet.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing all extracted data fields
        
    Raises:
        ParsingError: If document cannot be parsed
        DocumentProtectedError: If document is password protected
        ScannedDocumentError: If document is scanned and requires OCR
    """
    logger.info(f"Extracting data from MLS sheet: {pdf_path}")
    
    if is_document_protected(pdf_path):
        raise DocumentProtectedError(f"Document is password protected: {pdf_path}")
    
    if is_scanned_document(pdf_path):
        raise ScannedDocumentError(f"Document appears to be scanned: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            data = {}
            warnings = []
            
            # Extract overview/remarks
            try:
                remarks = extract_mls_overview(pdf_path)
                data['Remarks'] = normalize_text(remarks) if remarks else None
            except Exception as e:
                warnings.append(f"Failed to extract overview: {e}")
            
            # Extract other fields
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Extract BelowGroundSqFt
                below_ground_match = re.search(r'(?i)below\s*grade\s*sq\s*ft[:\s]+(\d+)', text)
                if below_ground_match:
                    data['BelowGroundSqFt'] = validate_sq_ft(below_ground_match.group(1), 'BelowGroundSqFt')
                
                # Extract Pool information
                pool_match = re.search(r'(?i)pool[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
                if pool_match:
                    data['PoolPresent'] = validate_yes_no(pool_match.group(1), 'PoolPresent')
                    if pool_match.group(2):
                        data['PoolType'] = normalize_text(pool_match.group(2))
                
                # Extract Fireplace information
                fireplace_match = re.search(r'(?i)fireplace[:\s]+(\d+)(?:\s*-\s*([^,\n]+))?', text)
                if fireplace_match:
                    data['FireplaceCount'] = validate_numeric(fireplace_match.group(1), 'FireplaceCount')
                    if fireplace_match.group(2):
                        data['FireplaceTypes'] = normalize_text(fireplace_match.group(2))
                
                # Extract FinishedBelowGroundSqFt
                finished_below_match = re.search(r'(?i)finished\s*below\s*grade\s*sq\s*ft[:\s]+(\d+)', text)
                if finished_below_match:
                    data['FinishedBelowGroundSqFt'] = validate_sq_ft(finished_below_match.group(1), 'FinishedBelowGroundSqFt')
                
                # Extract OriginalListPrice
                original_price_match = re.search(r'(?i)original\s*list\s*price[:\s]+\$?([\d,]+)', text)
                if original_price_match:
                    data['OriginalListPrice'] = validate_numeric(original_price_match.group(1), 'OriginalListPrice')
                
                # Extract BelowGradeBedroomCount and BelowGradeBathroomCount
                below_grade_match = re.search(r'(?i)below\s*grade\s*bedrooms[:\s]+(\d+)', text)
                if below_grade_match:
                    data['BelowGradeBedroomCount'] = validate_numeric(below_grade_match.group(1), 'BelowGradeBedroomCount')
                
                below_grade_bath_match = re.search(r'(?i)below\s*grade\s*bathrooms[:\s]+(\d+)', text)
                if below_grade_bath_match:
                    data['BelowGradeBathroomCount'] = validate_numeric(below_grade_bath_match.group(1), 'BelowGradeBathroomCount')
                
                # Extract PublicSewage
                sewage_match = re.search(r'(?i)sewer[:\s]+(public|private|septic)', text)
                if sewage_match:
                    data['PublicSewage'] = validate_enum(sewage_match.group(1), list(VALID_SEWER_TYPES), 'PublicSewage')
                
                # Extract PublicWater
                water_match = re.search(r'(?i)water[:\s]+(public|private|well)', text)
                if water_match:
                    data['PublicWater'] = validate_enum(water_match.group(1), list(VALID_WATER_TYPES), 'PublicWater')
                
                # Extract Fence information
                fence_match = re.search(r'(?i)fence[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
                if fence_match:
                    data['FencePresent'] = validate_yes_no(fence_match.group(1), 'FencePresent')
                    if fence_match.group(2):
                        data['FenceType'] = normalize_text(fence_match.group(2))
            
            # Log warnings for missing fields
            required_fields = [
                'BelowGroundSqFt', 'PoolPresent', 'PoolType', 'FireplaceCount',
                'FireplaceTypes', 'FinishedBelowGroundSqFt', 'OriginalListPrice',
                'BelowGradeBedroomCount', 'BelowGradeBathroomCount', 'PublicSewage',
                'PublicWater', 'FencePresent', 'FenceType', 'Remarks'
            ]
            
            missing = [field for field in required_fields if field not in data or data[field] is None]
            if missing:
                logger.warning(f"Missing required fields: {missing}")
                raise ValueError(f"Missing required fields: {missing}")
            
            return data
            
    except Exception as e:
        logger.error(f"Error extracting MLS sheet data: {e}")
        raise ParsingError(f"Failed to extract MLS sheet data: {e}")

def extract_tax_record_data(pdf_path: str) -> Dict[str, Any]:
    """
    Extract data from tax records.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted data
        
    Raises:
        ParsingError: If document cannot be parsed
        DocumentProtectedError: If document is password protected
        ScannedDocumentError: If document is scanned and requires OCR
    """
    logger.info(f"Extracting data from tax record: {pdf_path}")
    
    if is_document_protected(pdf_path):
        raise DocumentProtectedError(f"Document is password protected: {pdf_path}")
    
    if is_scanned_document(pdf_path):
        raise ScannedDocumentError(f"Document appears to be scanned: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            data = {}
            warnings = []
            
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Extract Tax Year
                tax_year_match = re.search(r'(?i)tax\s*year[:\s]+(\d{4})', text)
                if tax_year_match:
                    data['TaxYear'] = validate_numeric(tax_year_match.group(1), 'TaxYear')
                
                # Extract Property Value
                property_value_match = re.search(r'(?i)(?:total|assessed)\s*value[:\s]+\$?([\d,]+)', text)
                if property_value_match:
                    data['PropertyValue'] = validate_numeric(property_value_match.group(1), 'PropertyValue')
                
                # Extract Land Value
                land_value_match = re.search(r'(?i)land\s*value[:\s]+\$?([\d,]+)', text)
                if land_value_match:
                    data['LandValue'] = validate_numeric(land_value_match.group(1), 'LandValue')
                
                # Extract Improvement Value
                improvement_value_match = re.search(r'(?i)improvement\s*value[:\s]+\$?([\d,]+)', text)
                if improvement_value_match:
                    data['ImprovementValue'] = validate_numeric(improvement_value_match.group(1), 'ImprovementValue')
                
                # Extract Tax Amount
                tax_amount_match = re.search(r'(?i)total\s*tax[:\s]+\$?([\d,]+)', text)
                if tax_amount_match:
                    data['TaxAmount'] = validate_numeric(tax_amount_match.group(1), 'TaxAmount')
                
                # Extract Exemptions
                exemption_match = re.search(r'(?i)exemptions[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
                if exemption_match:
                    data['HasExemptions'] = validate_yes_no(exemption_match.group(1), 'HasExemptions')
                    if exemption_match.group(2):
                        data['ExemptionTypes'] = normalize_text(exemption_match.group(2))
                
                # Extract Last Sale Date
                sale_date_match = re.search(r'(?i)last\s*sale\s*date[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text)
                if sale_date_match:
                    data['LastSaleDate'] = validate_date(sale_date_match.group(1), 'LastSaleDate')
                
                # Extract Last Sale Price
                sale_price_match = re.search(r'(?i)last\s*sale\s*price[:\s]+\$?([\d,]+)', text)
                if sale_price_match:
                    data['LastSalePrice'] = validate_numeric(sale_price_match.group(1), 'LastSalePrice')
                
                # Extract Property Class
                property_class_match = re.search(r'(?i)property\s*class[:\s]+([^,\n]+)', text)
                if property_class_match:
                    data['PropertyClass'] = validate_enum(
                        property_class_match.group(1),
                        list(VALID_PROPERTY_CLASSES),
                        'PropertyClass'
                    )
                
                # Extract Zoning
                zoning_match = re.search(r'(?i)zoning[:\s]+([^,\n]+)', text)
                if zoning_match:
                    data['Zoning'] = validate_enum(
                        zoning_match.group(1),
                        list(VALID_ZONING_TYPES),
                        'Zoning'
                    )
            
            # Log warnings for missing fields
            required_fields = [
                'TaxYear', 'PropertyValue', 'LandValue', 'ImprovementValue',
                'TaxAmount', 'HasExemptions', 'LastSaleDate', 'LastSalePrice',
                'PropertyClass', 'Zoning'
            ]
            
            missing = [field for field in required_fields if field not in data or data[field] is None]
            if missing:
                logger.warning(f"Missing required fields: {missing}")
                raise ValueError(f"Missing required fields: {missing}")
            
            return data
            
    except Exception as e:
        logger.error(f"Error extracting tax record data: {e}")
        raise ParsingError(f"Failed to extract tax record data: {e}")

def parse_document(pdf_path: str) -> ParsingResult:
    """
    Main function to parse any supported document type.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        ParsingResult containing extracted data and metadata
        
    Raises:
        ParsingError: If document cannot be parsed
        DocumentProtectedError: If document is password protected
        ScannedDocumentError: If document is scanned and requires OCR
    """
    logger.info(f"Parsing document: {pdf_path}")
    
    try:
        # Check if document is protected
        if is_document_protected(pdf_path):
            raise DocumentProtectedError(f"Document is password protected: {pdf_path}")
        
        # Check if document is scanned
        is_scanned = is_scanned_document(pdf_path)
        
        # Determine document type
        if "mls" in pdf_path.lower():
            document_type = DocumentType.MLS_SHEET
            if is_scanned:
                # Use OCR for scanned MLS sheets
                text = extract_text_with_ocr(pdf_path)
                data = extract_mls_sheet_data_from_text(text)
            else:
                data = extract_mls_sheet_data(pdf_path)
        elif "tax" in pdf_path.lower():
            document_type = DocumentType.TAX_RECORD
            if is_scanned:
                # Use OCR for scanned tax records
                text = extract_text_with_ocr(pdf_path)
                data = extract_tax_record_data_from_text(text)
            else:
                data = extract_tax_record_data(pdf_path)
        else:
            document_type = DocumentType.SUBJECT_PRIOR
            if is_scanned:
                # Use OCR for scanned subject priors
                text = extract_text_with_ocr(pdf_path)
                data = extract_mls_sheet_data_from_text(text)
            else:
                data = extract_mls_sheet_data(pdf_path)
        
        return ParsingResult(
            data=data,
            document_type=document_type,
            warnings=[],
            errors=[],
            is_scanned=is_scanned,
            is_protected=False
        )
        
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise ParsingError(f"Failed to parse document: {e}")

def extract_mls_sheet_data_from_text(text: str) -> Dict[str, Any]:
    """
    Extract MLS sheet data from text (used for OCR results).
    """
    data = {}
    # Extract BelowGroundSqFt
    below_ground_match = re.search(r'(?i)below\s*grade\s*sq\s*ft[:\s]+(\d+)', text)
    if below_ground_match:
        data['BelowGroundSqFt'] = validate_sq_ft(below_ground_match.group(1), 'BelowGroundSqFt')
    pool_match = re.search(r'(?i)pool[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
    if pool_match:
        data['PoolPresent'] = validate_yes_no(pool_match.group(1), 'PoolPresent')
        if pool_match.group(2):
            data['PoolType'] = normalize_text(pool_match.group(2))
    fireplace_match = re.search(r'(?i)fireplace[:\s]+(\d+)(?:\s*-\s*([^,\n]+))?', text)
    if fireplace_match:
        data['FireplaceCount'] = validate_numeric(fireplace_match.group(1), 'FireplaceCount')
        if fireplace_match.group(2):
            data['FireplaceTypes'] = normalize_text(fireplace_match.group(2))
    finished_below_match = re.search(r'(?i)finished\s*below\s*grade\s*sq\s*ft[:\s]+(\d+)', text)
    if finished_below_match:
        data['FinishedBelowGroundSqFt'] = validate_sq_ft(finished_below_match.group(1), 'FinishedBelowGroundSqFt')
    original_price_match = re.search(r'(?i)original\s*list\s*price[:\s]+\$?([\d,]+)', text)
    if original_price_match:
        data['OriginalListPrice'] = validate_numeric(original_price_match.group(1), 'OriginalListPrice')
    below_grade_match = re.search(r'(?i)below\s*grade\s*bedrooms[:\s]+(\d+)', text)
    if below_grade_match:
        data['BelowGradeBedroomCount'] = validate_numeric(below_grade_match.group(1), 'BelowGradeBedroomCount')
    below_grade_bath_match = re.search(r'(?i)below\s*grade\s*bathrooms[:\s]+(\d+)', text)
    if below_grade_bath_match:
        data['BelowGradeBathroomCount'] = validate_numeric(below_grade_bath_match.group(1), 'BelowGradeBathroomCount')
    sewage_match = re.search(r'(?i)sewer[:\s]+(public|private|septic)', text)
    if sewage_match:
        data['PublicSewage'] = validate_enum(sewage_match.group(1), list(VALID_SEWER_TYPES), 'PublicSewage')
    water_match = re.search(r'(?i)water[:\s]+(public|private|well)', text)
    if water_match:
        data['PublicWater'] = validate_enum(water_match.group(1), list(VALID_WATER_TYPES), 'PublicWater')
    fence_match = re.search(r'(?i)fence[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
    if fence_match:
        data['FencePresent'] = validate_yes_no(fence_match.group(1), 'FencePresent')
        if fence_match.group(2):
            data['FenceType'] = normalize_text(fence_match.group(2))
    # Remarks extraction is not possible from OCR text in this mockup
    required_fields = [
        'BelowGroundSqFt', 'PoolPresent', 'FireplaceCount',
        'FinishedBelowGroundSqFt', 'OriginalListPrice',
        'BelowGradeBedroomCount', 'BelowGradeBathroomCount', 'PublicSewage',
        'PublicWater', 'FencePresent'
    ]
    missing = [field for field in required_fields if field not in data or data[field] is None]
    if missing:
        logger.warning(f"Missing required fields: {missing}")
        raise ValueError(f"Missing required fields: {missing}")
    return data

def extract_tax_record_data_from_text(text: str) -> Dict[str, Any]:
    data = {}
    # Extract Tax Year
    tax_year_match = re.search(r'(?i)tax\s*year[:\s]+(\d{4})', text)
    if tax_year_match:
        data['TaxYear'] = validate_numeric(tax_year_match.group(1), 'TaxYear')
    property_value_match = re.search(r'(?i)(?:total|assessed)\s*value[:\s]+\$?([\d,]+)', text)
    if property_value_match:
        data['PropertyValue'] = validate_numeric(property_value_match.group(1), 'PropertyValue')
    land_value_match = re.search(r'(?i)land\s*value[:\s]+\$?([\d,]+)', text)
    if land_value_match:
        data['LandValue'] = validate_numeric(land_value_match.group(1), 'LandValue')
    improvement_value_match = re.search(r'(?i)improvement\s*value[:\s]+\$?([\d,]+)', text)
    if improvement_value_match:
        data['ImprovementValue'] = validate_numeric(improvement_value_match.group(1), 'ImprovementValue')
    tax_amount_match = re.search(r'(?i)total\s*tax[:\s]+\$?([\d,]+)', text)
    if tax_amount_match:
        data['TaxAmount'] = validate_numeric(tax_amount_match.group(1), 'TaxAmount')
    exemption_match = re.search(r'(?i)exemptions[:\s]+(yes|no)(?:\s*-\s*([^,\n]+))?', text)
    if exemption_match:
        data['HasExemptions'] = validate_yes_no(exemption_match.group(1), 'HasExemptions')
        if exemption_match.group(2):
            data['ExemptionTypes'] = normalize_text(exemption_match.group(2))
    sale_date_match = re.search(r'(?i)last\s*sale\s*date[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text)
    if sale_date_match:
        data['LastSaleDate'] = validate_date(sale_date_match.group(1), 'LastSaleDate')
    sale_price_match = re.search(r'(?i)last\s*sale\s*price[:\s]+\$?([\d,]+)', text)
    if sale_price_match:
        data['LastSalePrice'] = validate_numeric(sale_price_match.group(1), 'LastSalePrice')
    property_class_match = re.search(r'(?i)property\s*class[:\s]+([^,\n]+)', text)
    if property_class_match:
        data['PropertyClass'] = validate_enum(
            property_class_match.group(1),
            list(VALID_PROPERTY_CLASSES),
            'PropertyClass'
        )
    zoning_match = re.search(r'(?i)zoning[:\s]+([^,\n]+)', text)
    if zoning_match:
        data['Zoning'] = validate_enum(
            zoning_match.group(1),
            list(VALID_ZONING_TYPES),
            'Zoning'
        )
    required_fields = [
        'TaxYear', 'PropertyValue', 'LandValue', 'ImprovementValue',
        'TaxAmount', 'HasExemptions', 'LastSaleDate', 'LastSalePrice',
        'PropertyClass', 'Zoning'
    ]
    missing = [field for field in required_fields if field not in data or data[field] is None]
    if missing:
        logger.warning(f"Missing required fields: {missing}")
        raise ValueError(f"Missing required fields: {missing}")
    return data 