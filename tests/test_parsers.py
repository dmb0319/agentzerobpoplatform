"""
Test suite for the parsers module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pdfplumber
from bpo_agent_tools.utils.parsers import (
    DocumentType,
    ParsingResult,
    ParsingError,
    DocumentProtectedError,
    ScannedDocumentError,
    is_document_protected,
    is_scanned_document,
    extract_mls_overview,
    extract_mls_sheet_data,
    extract_tax_record_data,
    parse_document,
    extract_text_with_ocr,
    extract_tax_record_data_from_text,
    normalize_text,
    validate_numeric,
    validate_sq_ft,
    validate_yes_no,
    validate_date,
    validate_enum,
    VALID_SEWER_TYPES,
    VALID_WATER_TYPES,
    VALID_PROPERTY_CLASSES,
    VALID_ZONING_TYPES
)

# Test data
SAMPLE_MLS_TEXT = """
Property Overview:
Beautiful 3 bed, 2 bath home in quiet neighborhood.
Recently renovated kitchen and bathrooms.
Large backyard with pool - in-ground concrete.

Features:
- Below Grade Sq Ft: 1200
- Pool: Yes - In-ground concrete
- Fireplace: 2 - Wood burning
- Finished Below Grade Sq Ft: 800
- Original List Price: $450,000
- Below Grade Bedrooms: 1
- Below Grade Bathrooms: 1
- Sewer: Public
- Water: Public
- Fence: Yes - Wood privacy

Room Information:
...
"""

# Test data for tax records
SAMPLE_TAX_TEXT = """
Tax Year: 2023
Total Value: $450,000
Land Value: $150,000
Improvement Value: $300,000
Total Tax: $12,500
Exemptions: Yes - Homestead
Last Sale Date: 01/15/2022
Last Sale Price: $425,000
Property Class: Residential
Zoning: R-1
"""

@pytest.fixture
def mock_pdf():
    """Create a mock PDF object for testing."""
    mock = MagicMock()
    mock.pages = [MagicMock()]
    mock.pages[0].extract_text.return_value = SAMPLE_MLS_TEXT
    return mock

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("Sample PDF content")
    return str(pdf_path)

def test_is_document_protected():
    """Test document protection detection."""
    with patch('pdfplumber.open') as mock_open:
        # Test non-protected document
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_open.return_value.__enter__.return_value = mock_pdf
        assert not is_document_protected("test.pdf")
        
        # Test protected document
        mock_open.side_effect = Exception("encrypted")
        assert is_document_protected("test.pdf")
        
        # Test other error
        mock_open.side_effect = Exception("other error")
        with pytest.raises(Exception):
            is_document_protected("test.pdf")

def test_is_scanned_document():
    """Test scanned document detection."""
    with patch('pdfplumber.open') as mock_open:
        # Test non-scanned document
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "Sample text" * 10
        mock_open.return_value.__enter__.return_value = mock_pdf
        assert not is_scanned_document("test.pdf")
        
        # Test scanned document
        mock_pdf.pages[0].extract_text.return_value = ""
        assert is_scanned_document("test.pdf")

def test_extract_mls_overview():
    """Test MLS overview extraction."""
    with patch('pdfplumber.open') as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = SAMPLE_MLS_TEXT
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        overview = extract_mls_overview("test.pdf")
        assert overview is not None
        assert "Beautiful 3 bed" in overview
        assert "Recently renovated" in overview
        assert "Large backyard" in overview

def test_extract_mls_sheet_data():
    """Test MLS sheet data extraction."""
    with patch('pdfplumber.open') as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = SAMPLE_MLS_TEXT
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        data = extract_mls_sheet_data("test.pdf")
        
        # Verify extracted fields
        assert data['BelowGroundSqFt'] == 1200
        assert data['PoolPresent'] is True
        assert data['PoolType'] == "In-ground concrete"
        assert data['FireplaceCount'] == 2
        assert data['FireplaceTypes'] == "Wood burning"
        assert data['FinishedBelowGroundSqFt'] == 800
        assert data['OriginalListPrice'] == 450000
        assert data['BelowGradeBedroomCount'] == 1
        assert data['BelowGradeBathroomCount'] == 1
        assert data['PublicSewage'] == "Public"
        assert data['PublicWater'] == "Public"
        assert data['FencePresent'] is True
        assert data['FenceType'] == "Wood privacy"

def test_extract_tax_record_data():
    """Test tax record data extraction."""
    with patch('pdfplumber.open') as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = SAMPLE_TAX_TEXT
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        data = extract_tax_record_data("test.pdf")
        
        # Verify extracted fields
        assert data['TaxYear'] == 2023
        assert data['PropertyValue'] == 450000
        assert data['LandValue'] == 150000
        assert data['ImprovementValue'] == 300000
        assert data['TaxAmount'] == 12500
        assert data['HasExemptions'] is True
        assert data['ExemptionTypes'] == "Homestead"
        assert data['LastSaleDate'] == "01/15/2022"
        assert data['LastSalePrice'] == 425000
        assert data['PropertyClass'] == "Residential"
        assert data['Zoning'] == "R-1"

def test_parse_document():
    """Test main document parsing function."""
    with patch('pdfplumber.open') as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = SAMPLE_MLS_TEXT
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        # Test MLS sheet
        result = parse_document("mls_sheet.pdf")
        assert isinstance(result, ParsingResult)
        assert result.document_type == DocumentType.MLS_SHEET
        assert not result.is_scanned
        assert not result.is_protected
        
        # Test tax record
        result = parse_document("tax_record.pdf")
        assert isinstance(result, ParsingResult)
        assert result.document_type == DocumentType.TAX_RECORD
        
        # Test subject prior
        result = parse_document("subject_prior.pdf")
        assert isinstance(result, ParsingResult)
        assert result.document_type == DocumentType.SUBJECT_PRIOR

def test_error_handling():
    """Test error handling in various scenarios."""
    with patch('pdfplumber.open') as mock_open:
        # Test protected document
        mock_open.side_effect = Exception("encrypted")
        with pytest.raises(DocumentProtectedError):
            parse_document("test.pdf")
        
        # Test scanned document
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = ""
        mock_open.side_effect = None
        mock_open.return_value.__enter__.return_value = mock_pdf
        with pytest.raises(ScannedDocumentError):
            parse_document("test.pdf")
        
        # Test general parsing error
        mock_open.side_effect = Exception("general error")
        with pytest.raises(ParsingError):
            parse_document("test.pdf")

def test_missing_fields():
    """Test handling of missing fields in MLS sheet."""
    with patch('pdfplumber.open') as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "Minimal MLS data"
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        data = extract_mls_sheet_data("test.pdf")
        assert isinstance(data, dict)
        # Verify that missing fields are handled gracefully
        assert 'BelowGroundSqFt' not in data
        assert 'PoolPresent' not in data

def test_extract_text_with_ocr():
    """Test OCR text extraction."""
    with patch('pdfplumber.open') as mock_open, \
         patch('pytesseract.image_to_string') as mock_ocr:
        # Mock PDF and OCR
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_image = MagicMock()
        mock_page.to_image.return_value = mock_image
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf
        mock_ocr.return_value = "Sample OCR text"
        
        text = extract_text_with_ocr("test.pdf")
        assert text == "Sample OCR text"
        assert mock_ocr.called

def test_extract_tax_record_data_from_text():
    """Test tax record data extraction from OCR text."""
    data = extract_tax_record_data_from_text(SAMPLE_TAX_TEXT)
    
    # Verify extracted fields
    assert data['TaxYear'] == 2023
    assert data['PropertyValue'] == 450000
    assert data['LandValue'] == 150000
    assert data['ImprovementValue'] == 300000
    assert data['TaxAmount'] == 12500
    assert data['HasExemptions'] is True
    assert data['ExemptionTypes'] == "Homestead"
    assert data['LastSaleDate'] == "01/15/2022"
    assert data['LastSalePrice'] == 425000
    assert data['PropertyClass'] == "Residential"
    assert data['Zoning'] == "R-1"

def test_parse_document_with_ocr():
    """Test document parsing with OCR for scanned documents."""
    with patch('pdfplumber.open') as mock_open, \
         patch('pytesseract.image_to_string') as mock_ocr:
        # Mock PDF and OCR
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_image = MagicMock()
        mock_page.to_image.return_value = mock_image
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        # Test scanned MLS sheet
        mock_page.extract_text.return_value = ""  # Empty text indicates scanned document
        mock_ocr.return_value = SAMPLE_MLS_TEXT
        result = parse_document("mls_sheet.pdf")
        assert result.is_scanned
        assert result.document_type == DocumentType.MLS_SHEET
        
        # Test scanned tax record
        mock_ocr.return_value = SAMPLE_TAX_TEXT
        result = parse_document("tax_record.pdf")
        assert result.is_scanned
        assert result.document_type == DocumentType.TAX_RECORD

def test_ocr_error_handling():
    """Test error handling in OCR process."""
    with patch('pdfplumber.open') as mock_open, \
         patch('pytesseract.image_to_string') as mock_ocr:
        # Mock OCR failure
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_image = MagicMock()
        mock_page.to_image.return_value = mock_image
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf
        mock_ocr.side_effect = Exception("OCR failed")
        
        with pytest.raises(ParsingError):
            extract_text_with_ocr("test.pdf")

def test_validation_utilities():
    """Test the validation utility functions."""
    # Test normalize_text
    assert normalize_text("  Test  String  ") == "Test String"
    assert normalize_text("Test\nString") == "Test String"
    assert normalize_text(None) == ""  # Empty string is the expected behavior

    # Test validate_numeric
    assert validate_numeric("1,234", "test") == 1234
    assert validate_numeric("0", "test") == 0
    with pytest.raises(ValueError):
        validate_numeric("abc", "test")
    with pytest.raises(ValueError):
        validate_numeric("", "test")

    # Test validate_sq_ft
    assert validate_sq_ft("1,234 sq ft", "test") == 1234
    assert validate_sq_ft("0", "test") == 0
    with pytest.raises(ValueError):
        validate_sq_ft("abc", "test")
    with pytest.raises(ValueError):
        validate_sq_ft("", "test")

    # Test validate_yes_no
    assert validate_yes_no("yes", "test") is True
    assert validate_yes_no("no", "test") is False
    assert validate_yes_no("YES", "test") is True
    assert validate_yes_no("NO", "test") is False
    with pytest.raises(ValueError):
        validate_yes_no("maybe", "test")
    with pytest.raises(ValueError):
        validate_yes_no("", "test")

    # Test validate_date
    assert validate_date("01/15/2022", "test") == "01/15/2022"
    with pytest.raises(ValueError):
        validate_date("invalid", "test")
    with pytest.raises(ValueError):
        validate_date("", "test")

    # Test validate_enum
    valid_values = ["Residential", "Commercial", "Industrial"]
    assert validate_enum("residential", valid_values, "test") == "Residential"
    assert validate_enum("RESIDENTIAL", valid_values, "test") == "Residential"
    with pytest.raises(ValueError):
        validate_enum("invalid", valid_values, "test")
    with pytest.raises(ValueError):
        validate_enum("", valid_values, "test")

def test_mls_sheet_data_validation():
    """Test validation in MLS sheet data extraction."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = """
    Below Grade Sq Ft: 1,234 sq ft
    Pool: Yes - Inground
    Fireplace: 2 - Wood Burning
    Finished Below Grade Sq Ft: 500 sq ft
    Original List Price: $500,000
    Below Grade Bedrooms: 2
    Below Grade Bathrooms: 1
    Sewer: Public
    Water: Private
    Fence: Yes - Wood
    """

    with patch('pdfplumber.open', return_value=mock_pdf):
        with patch('bpo_agent_tools.utils.parsers.is_scanned_document', return_value=False):
            data = extract_mls_sheet_data("test.pdf")
            assert data['BelowGroundSqFt'] == 1234
            assert data['PoolPresent'] is True
            assert data['FireplaceCount'] == 2
            assert data['FinishedBelowGroundSqFt'] == 500
            assert data['OriginalListPrice'] == 500000
            assert data['BelowGradeBedroomCount'] == 2
            assert data['BelowGradeBathroomCount'] == 1
            assert data['PublicSewage'] == "Public"
            assert data['PublicWater'] == "Private"
            assert data['FencePresent'] is True

def test_tax_record_data_validation():
    """Test validation in tax record data extraction."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = """
    Tax Year: 2024
    Total Value: $500,000
    Land Value: $200,000
    Improvement Value: $300,000
    Total Tax: $10,000
    Exemptions: Yes - Homestead
    Last Sale Date: 01/01/2024
    Last Sale Price: $450,000
    Property Class: Residential
    Zoning: R1
    """

    with patch('pdfplumber.open', return_value=mock_pdf):
        with patch('bpo_agent_tools.utils.parsers.is_scanned_document', return_value=False):
            data = extract_tax_record_data("test.pdf")
            assert data['TaxYear'] == 2024
            assert data['PropertyValue'] == 500000
            assert data['LandValue'] == 200000
            assert data['ImprovementValue'] == 300000
            assert data['TaxAmount'] == 10000
            assert data['HasExemptions'] is True
            assert data['ExemptionTypes'] == "Homestead"
            assert data['LastSaleDate'] == "01/01/2024"
            assert data['LastSalePrice'] == 450000
            assert data['PropertyClass'] == "Residential"
            assert data['Zoning'] == "R1"

def test_invalid_data_handling():
    """Test handling of invalid data in extraction functions."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = """
    Below Grade Sq Ft: invalid
    Pool: maybe
    Fireplace: -1
    Finished Below Grade Sq Ft: abc
    Original List Price: $invalid
    Below Grade Bedrooms: -2
    Below Grade Bathrooms: invalid
    Sewer: invalid
    Water: invalid
    Fence: maybe
    """

    with patch('pdfplumber.open', return_value=mock_pdf):
        with patch('bpo_agent_tools.utils.parsers.is_scanned_document', return_value=False):
            with pytest.raises(ParsingError):
                extract_mls_sheet_data("test.pdf")

def test_invalid_tax_data_handling():
    """Test handling of invalid data in tax record extraction."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = """
    Tax Year: invalid
    Total Value: $invalid
    Land Value: $invalid
    Improvement Value: $invalid
    Total Tax: $invalid
    Exemptions: maybe
    Last Sale Date: invalid
    Last Sale Price: $invalid
    Property Class: invalid
    Zoning: invalid
    """

    with patch('pdfplumber.open', return_value=mock_pdf):
        with patch('bpo_agent_tools.utils.parsers.is_scanned_document', return_value=False):
            with pytest.raises(ParsingError):
                extract_tax_record_data("test.pdf")

def test_mixed_case_handling():
    """Test handling of mixed case values in extraction."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock()]
    mock_pdf.pages[0].extract_text.return_value = """
    Pool: YES - INGROUND
    Fireplace: 2 - WOOD BURNING
    Sewer: PUBLIC
    Water: PRIVATE
    Fence: YES - WOOD
    Property Class: RESIDENTIAL
    Zoning: R-1
    """

    with patch('pdfplumber.open', return_value=mock_pdf):
        with patch('bpo_agent_tools.utils.parsers.is_scanned_document', return_value=False):
            data = extract_mls_sheet_data("test.pdf")
            assert data['PoolPresent'] is True
            assert data['FireplaceCount'] == 2
            assert data['PublicSewage'] == "Public"
            assert data['PublicWater'] == "Private"
            assert data['FencePresent'] is True
            assert data['PropertyClass'] == "Residential"
            assert data['Zoning'] == "R-1" 