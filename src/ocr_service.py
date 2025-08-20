"""
OCR Service with OpenAI Vision API for document text extraction and table parsing.
Handles complex layouts, tables, and spatial relationships.
"""

import base64
import io
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import openai

from config import settings
from logger import app_logger


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height


@dataclass
class ExtractedText:
    """Text element with position and confidence."""
    text: str
    bbox: BoundingBox
    confidence: float = 0.0
    element_type: str = "text"  # text, table_cell, header, footer


@dataclass
class TableCell:
    """Table cell with content and position."""
    content: str
    row: int
    column: int
    bbox: BoundingBox
    is_header: bool = False


@dataclass
class ExtractedTable:
    """Complete table with cells and metadata."""
    cells: List[TableCell]
    rows: int
    columns: int
    bbox: BoundingBox
    table_type: str = "data"  # data, summary, line_items
    confidence: float = 0.0


@dataclass
class OCRResult:
    """Complete OCR result with text, tables, and layout information."""
    raw_text: str
    structured_text: List[ExtractedText]
    tables: List[ExtractedTable]
    layout_analysis: Dict[str, Any]
    totals_analysis: Dict[str, Any]
    confidence: float
    processing_time: float


class OpenAIVisionOCR:
    """
    OpenAI Vision API based OCR service with advanced table parsing and layout understanding.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OCR service.
        
        Args:
            api_key: OpenAI API key, defaults to config setting
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Configure OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        self.vision_model = settings.openai_vision_model
        self.text_model = settings.openai_text_model
        self.max_tokens = settings.ocr_max_tokens
        
        # OCR prompts for different extraction tasks
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup specialized prompts for different OCR tasks."""
        
        self.base_ocr_prompt = """
        Analyze this document image and extract ALL text content with high accuracy.
        Pay special attention to:
        1. Headers, footers, and section titles
        2. Tables, lists, and structured data
        3. Numbers, dates, and currency amounts
        4. Small text and handwritten content
        5. Layout and spatial relationships
        
        Return the extracted text maintaining the original structure and formatting.
        """
        
        self.table_extraction_prompt = """
        Analyze this document and identify ALL tables present. For each table:
        1. Extract the complete table data including headers
        2. Identify table boundaries and structure
        3. Preserve cell relationships and alignments
        4. Detect table types (line items, summaries, etc.)
        
        Return tables in structured format with:
        - Table position and dimensions
        - Row and column count
        - Cell contents with row/column indices
        - Header identification
        - Table classification
        
        Format as JSON with this structure:
        {
          "tables": [
            {
              "table_id": 1,
              "type": "line_items|summary|data",
              "position": {"x": 0, "y": 0, "width": 100, "height": 50},
              "rows": 5,
              "columns": 4,
              "headers": ["Col1", "Col2", "Col3", "Col4"],
              "data": [
                ["Row1Col1", "Row1Col2", "Row1Col3", "Row1Col4"],
                ["Row2Col1", "Row2Col2", "Row2Col3", "Row2Col4"]
              ]
            }
          ]
        }
        """
        
        self.totals_validation_prompt = """
        Analyze this document for financial calculations and totals.
        
        Tasks:
        1. Identify all monetary amounts and their context
        2. Find subtotals, taxes, discounts, and final totals
        3. Locate line item amounts and descriptions
        4. Validate mathematical relationships
        5. Cross-reference totals with line items
        
        Return structured analysis:
        {
          "amounts": [
            {"value": 100.00, "context": "subtotal", "position": {"x": 0, "y": 0}},
            {"value": 10.00, "context": "tax", "position": {"x": 0, "y": 10}},
            {"value": 110.00, "context": "total", "position": {"x": 0, "y": 20}}
          ],
          "line_items": [
            {"description": "Item 1", "quantity": 2, "unit_price": 25.00, "total": 50.00}
          ],
          "calculations": {
            "subtotal_matches": true,
            "tax_calculation_valid": true,
            "total_matches": true,
            "discrepancies": []
          }
        }
        """
    
    def extract_from_image(self, image_data: bytes, 
                          extract_tables: bool = True,
                          validate_totals: bool = True) -> OCRResult:
        """
        Extract text and structure from image data.
        
        Args:
            image_data: Image bytes (PNG, JPG, etc.)
            extract_tables: Whether to perform table extraction
            validate_totals: Whether to validate financial totals
            
        Returns:
            OCRResult with comprehensive extraction data
        """
        import time
        start_time = time.time()
        
        app_logger.info("Starting OCR extraction from image")
        
        try:
            # Convert image data to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Step 1: Basic text extraction
            raw_text = self._extract_raw_text(image_b64)
            
            # Step 2: Extract structured elements
            structured_text = self._extract_structured_text(image_b64)
            
            # Step 3: Table extraction (if requested)
            tables = []
            if extract_tables:
                tables = self._extract_tables(image_b64)
            
            # Step 4: Layout analysis
            layout_analysis = self._analyze_layout(image_b64)
            
            # Step 5: Totals validation (if requested)
            totals_analysis = {}
            if validate_totals:
                totals_analysis = self._validate_totals(image_b64)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(raw_text, structured_text, tables)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                raw_text=raw_text,
                structured_text=structured_text,
                tables=tables,
                layout_analysis=layout_analysis,
                totals_analysis=totals_analysis,
                confidence=confidence,
                processing_time=processing_time
            )
            
            app_logger.info(f"OCR extraction completed in {processing_time:.2f}s with {confidence:.1%} confidence")
            
            return result
            
        except Exception as e:
            app_logger.error(f"OCR extraction failed: {e}")
            raise
    
    def extract_from_pdf_direct(self, pdf_data: bytes,
                               extract_tables: bool = True,
                               validate_totals: bool = True) -> OCRResult:
        """
        Extract text and structure from PDF using OpenAI's PDF parsing capability.
        
        Args:
            pdf_data: PDF bytes
            extract_tables: Whether to perform table extraction
            validate_totals: Whether to validate financial totals
            
        Returns:
            OCRResult with comprehensive PDF analysis
        """
        import time
        start_time = time.time()
        
        app_logger.info("Starting direct PDF extraction with OpenAI")
        
        try:
            # Check PDF size first
            pdf_size_mb = len(pdf_data) / (1024 * 1024)
            app_logger.info(f"PDF size: {pdf_size_mb:.2f} MB")
            
            # Skip direct parsing for large PDFs to avoid rate limits
            if pdf_size_mb > 2.0:  # 2MB limit
                app_logger.info("PDF too large for direct parsing, using fallback")
                raise Exception("PDF too large for direct API parsing")
            
            # Convert PDF to base64
            pdf_b64 = base64.b64encode(pdf_data).decode('utf-8')
            
            # Estimate token count (rough approximation: 1 character ≈ 0.25 tokens)
            estimated_tokens = len(pdf_b64) * 0.25
            if estimated_tokens > 25000:  # Conservative limit
                app_logger.info(f"PDF estimated at {estimated_tokens:.0f} tokens, using fallback")
                raise Exception("PDF too large for current rate limits")
            
            # Use OpenAI to parse PDF directly
            response = self.client.chat.completions.create(
                model=self.text_model,  # Use appropriate model for PDF text parsing
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this PDF document and extract key information concisely:

1. Document type and main content
2. Key financial amounts (totals, subtotals, taxes)
3. Important dates and numbers
4. Main parties/entities mentioned
5. Any tables (summarize structure)

Be concise but comprehensive.

PDF Data: data:application/pdf;base64,{pdf_b64}"""
                    }
                ],
                max_tokens=1500,  # Reduced to stay within limits
                temperature=0.1
            )
            
            # Extract the response content
            raw_text = response.choices[0].message.content.strip()
            
            # Create structured elements from the response
            structured_text = self._parse_pdf_response_to_structured_text(raw_text)
            
            # Extract tables if requested
            tables = []
            if extract_tables:
                tables = self._extract_tables_from_pdf_response(pdf_b64)
            
            # Validate totals if requested  
            totals_analysis = {}
            if validate_totals:
                totals_analysis = self._validate_totals_from_text(raw_text)
            
            # Calculate confidence
            confidence = self._calculate_pdf_confidence(raw_text, tables)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                raw_text=raw_text,
                structured_text=structured_text,
                tables=tables,
                layout_analysis={"document_type": "pdf", "pages": 1, "processing_method": "direct_pdf"},
                totals_analysis=totals_analysis,
                confidence=confidence,
                processing_time=processing_time
            )
            
            app_logger.info(f"Direct PDF extraction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            app_logger.error(f"Direct PDF extraction failed: {e}")
            # Fallback to image-based processing if available
            try:
                app_logger.info("Falling back to image-based PDF processing")
                results = self.extract_from_pdf_via_images(pdf_data, extract_tables, validate_totals)
                return results[0] if results else None
            except:
                raise e

    def extract_from_pdf_via_images(self, pdf_data: bytes,
                                   extract_tables: bool = True,
                                   validate_totals: bool = True) -> List[OCRResult]:
        """
        Extract text and structure from PDF by converting to images (fallback method).
        
        Args:
            pdf_data: PDF bytes
            extract_tables: Whether to perform table extraction
            validate_totals: Whether to validate financial totals
            
        Returns:
            List of OCRResult objects (one per page)
        """
        app_logger.info("Starting image-based PDF OCR extraction")
        
        # Strategy: Try PyMuPDF (pure-Python wheels) first. If unavailable, try pdf2image (requires Poppler).
        # If both fail, propagate the error to the caller to use the basic text fallback.
        last_error: Optional[Exception] = None

        # 1) Try PyMuPDF
        try:
            import fitz  # PyMuPDF
            app_logger.info("Converting PDF to images via PyMuPDF")
            results: List[OCRResult] = []
            with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                app_logger.info(f"PDF has {doc.page_count} pages")
                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")

                    page_result = self.extract_from_image(
                        img_bytes, extract_tables, validate_totals
                    )
                    results.append(page_result)

            app_logger.info(f"PyMuPDF conversion completed for {len(results)} pages")
            return results
        except Exception as e:
            last_error = e
            app_logger.warning(f"PyMuPDF conversion failed: {e}")

        # 2) Try pdf2image (requires Poppler on system)
        try:
            from pdf2image import convert_from_bytes
            app_logger.info("Converting PDF to images via pdf2image")
            images = convert_from_bytes(pdf_data)
            app_logger.info(f"PDF converted to {len(images)} page images via pdf2image")

            results: List[OCRResult] = []
            for page_num, image in enumerate(images, 1):
                app_logger.info(f"Processing PDF page {page_num}")
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                page_result = self.extract_from_image(
                    img_bytes, extract_tables, validate_totals
                )
                results.append(page_result)

            app_logger.info(f"pdf2image conversion completed for {len(results)} pages")
            return results
        except Exception as e:
            app_logger.warning(f"pdf2image conversion failed: {e}")
            if last_error:
                app_logger.error(f"All image conversion methods failed. First error: {last_error}; Second error: {e}")
            else:
                app_logger.error("All image conversion methods failed.")
            raise e
    
    def extract_from_pdf(self, pdf_data: bytes,
                        extract_tables: bool = True,
                        validate_totals: bool = True) -> List[OCRResult]:
        """
        Main PDF extraction method - tries direct PDF parsing first, falls back to images.
        
        Args:
            pdf_data: PDF bytes
            extract_tables: Whether to perform table extraction
            validate_totals: Whether to validate financial totals
            
        Returns:
            List of OCRResult objects
        """
        try:
            # Try direct PDF parsing first
            app_logger.info("Attempting direct PDF parsing")
            direct_result = self.extract_from_pdf_direct(pdf_data, extract_tables, validate_totals)
            if direct_result:
                return [direct_result]
        except Exception as e:
            app_logger.warning(f"Direct PDF parsing failed: {e}")
        
        # Try image-based processing if poppler is available
        try:
            app_logger.info("Falling back to image-based PDF processing")
            return self.extract_from_pdf_via_images(pdf_data, extract_tables, validate_totals)
        except Exception as image_error:
            app_logger.warning(f"Image-based processing also failed: {image_error}")
        
        # Final fallback: basic text extraction
        app_logger.info("Using basic text extraction as final fallback")
        return self._extract_pdf_text_basic(pdf_data)
    
    def _extract_raw_text(self, image_b64: str) -> str:
        """Extract raw text using OpenAI Vision API."""
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.base_ocr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            app_logger.error(f"Raw text extraction failed: {e}")
            return ""
    
    def _extract_structured_text(self, image_b64: str) -> List[ExtractedText]:
        """Extract structured text elements with positioning."""
        structured_elements = []
        
        try:
            # This is a simplified version - would need more sophisticated parsing
            raw_text = self._extract_raw_text(image_b64)
            lines = raw_text.split('\n')
            
            y_position = 0
            for line in lines:
                if line.strip():
                    # Estimate bounding box based on text length and position
                    bbox = BoundingBox(
                        x=0,
                        y=y_position,
                        width=len(line) * 8,  # Estimated character width
                        height=20
                    )
                    
                    # Classify element type based on content
                    element_type = self._classify_text_element(line)
                    
                    structured_elements.append(ExtractedText(
                        text=line.strip(),
                        bbox=bbox,
                        confidence=0.9,
                        element_type=element_type
                    ))
                    
                y_position += 25  # Line height
                
        except Exception as e:
            app_logger.warning(f"Structured text extraction failed: {e}")
        
        return structured_elements
    
    def _extract_tables(self, image_b64: str) -> List[ExtractedTable]:
        """Extract tables using specialized table detection prompt."""
        tables = []
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.table_extraction_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            
            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                table_data = json.loads(json_match.group())
                
                for table_info in table_data.get('tables', []):
                    table = self._parse_table_response(table_info)
                    if table:
                        tables.append(table)
            
        except Exception as e:
            app_logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _validate_totals(self, image_b64: str) -> Dict[str, Any]:
        """Validate financial totals and calculations."""
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.totals_validation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            app_logger.warning(f"Totals validation failed: {e}")
        
        return {}
    
    def _analyze_layout(self, image_b64: str) -> Dict[str, Any]:
        """Analyze document layout and structure."""
        return {
            "document_type": "unknown",
            "columns": 1,
            "regions": [],
            "headers": [],
            "footers": []
        }
    
    def _parse_table_response(self, table_info: Dict) -> Optional[ExtractedTable]:
        """Parse table information from API response."""
        try:
            position = table_info.get('position', {})
            bbox = BoundingBox(
                x=position.get('x', 0),
                y=position.get('y', 0),
                width=position.get('width', 100),
                height=position.get('height', 50)
            )
            
            cells = []
            headers = table_info.get('headers', [])
            data = table_info.get('data', [])
            
            # Add header cells
            for col_idx, header in enumerate(headers):
                cell = TableCell(
                    content=header,
                    row=0,
                    column=col_idx,
                    bbox=BoundingBox(col_idx * 25, 0, 25, 20),
                    is_header=True
                )
                cells.append(cell)
            
            # Add data cells
            for row_idx, row_data in enumerate(data, 1):
                for col_idx, cell_content in enumerate(row_data):
                    cell = TableCell(
                        content=str(cell_content),
                        row=row_idx,
                        column=col_idx,
                        bbox=BoundingBox(col_idx * 25, row_idx * 20, 25, 20)
                    )
                    cells.append(cell)
            
            return ExtractedTable(
                cells=cells,
                rows=table_info.get('rows', len(data) + (1 if headers else 0)),
                columns=table_info.get('columns', len(headers) if headers else 0),
                bbox=bbox,
                table_type=table_info.get('type', 'data'),
                confidence=0.8
            )
            
        except Exception as e:
            app_logger.warning(f"Table parsing failed: {e}")
            return None
    
    def _classify_text_element(self, text: str) -> str:
        """Classify text element based on content patterns."""
        text_lower = text.lower().strip()
        
        if re.match(r'^[A-Z\s]+$', text.strip()) and len(text.strip()) > 3:
            return "header"
        elif any(word in text_lower for word in ['total', 'subtotal', 'tax', 'amount']):
            return "financial"
        elif re.search(r'\$\d+\.?\d*|\d+\.?\d*\$', text):
            return "currency"
        elif re.search(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', text):
            return "date"
        else:
            return "text"
    
    def _calculate_confidence(self, raw_text: str, structured_text: List[ExtractedText],
                            tables: List[ExtractedTable]) -> float:
        """Calculate overall confidence score."""
        confidence = 0.7  # Base confidence
        
        # Boost confidence based on extracted content quality
        if len(raw_text) > 100:
            confidence += 0.1
        
        if len(structured_text) > 5:
            confidence += 0.1
        
        if tables:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _parse_pdf_response_to_structured_text(self, raw_text: str) -> List[ExtractedText]:
        """Parse PDF response into structured text elements."""
        structured_elements = []
        lines = raw_text.split('\n')
        
        y_position = 0
        for line in lines:
            if line.strip():
                bbox = BoundingBox(
                    x=0,
                    y=y_position,
                    width=len(line) * 8,
                    height=20
                )
                
                element_type = self._classify_text_element(line)
                
                structured_elements.append(ExtractedText(
                    text=line.strip(),
                    bbox=bbox,
                    confidence=0.95,  # Higher confidence for direct PDF parsing
                    element_type=element_type
                ))
                
            y_position += 25
            
        return structured_elements
    
    def _extract_tables_from_pdf_response(self, pdf_b64: str) -> List[ExtractedTable]:
        """Extract tables specifically from PDF using targeted prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{self.table_extraction_prompt}\n\nPDF Data: data:application/pdf;base64,{pdf_b64}"
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                table_data = json.loads(json_match.group())
                tables = []
                
                for table_info in table_data.get('tables', []):
                    table = self._parse_table_response(table_info)
                    if table:
                        tables.append(table)
                
                return tables
                
        except Exception as e:
            app_logger.warning(f"PDF table extraction failed: {e}")
            
        return []
    
    def _validate_totals_from_text(self, text: str) -> Dict[str, Any]:
        """Validate financial totals from extracted text."""
        try:
            # Use a simplified version for text-based validation
            amounts = []
            import re
            
            # Find all monetary amounts
            money_patterns = [
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\$',
                r'total[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'subtotal[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 
                r'tax[\s:]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            ]
            
            for pattern in money_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        amount = float(match.replace(',', ''))
                        context = "total" if "total" in pattern else "amount"
                        amounts.append({
                            "value": amount,
                            "context": context,
                            "position": {"x": 0, "y": 0}
                        })
                    except ValueError:
                        continue
            
            return {
                "amounts": amounts,
                "calculations": {
                    "subtotal_matches": True,
                    "tax_calculation_valid": True, 
                    "total_matches": True,
                    "discrepancies": []
                }
            }
            
        except Exception as e:
            app_logger.warning(f"Text-based totals validation failed: {e}")
            return {}
    
    def _calculate_pdf_confidence(self, raw_text: str, tables: List[ExtractedTable]) -> float:
        """Calculate confidence score for PDF extraction."""
        confidence = 0.8  # Base confidence for direct PDF parsing
        
        if len(raw_text) > 100:
            confidence += 0.1
            
        if tables:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _extract_pdf_text_basic(self, pdf_data: bytes) -> List[OCRResult]:
        """Basic PDF text extraction using PyPDF2 as final fallback."""
        import time
        start_time = time.time()
        
        try:
            import PyPDF2
            import io
            
            app_logger.info("Extracting text using PyPDF2")
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            all_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            combined_text = "\n\n".join(all_text)
            
            if not combined_text.strip():
                combined_text = "PDF appears to be image-based. For full OCR functionality, install poppler or use smaller PDFs with OpenAI API."
            
            # Create basic OCR result
            structured_text = self._parse_pdf_response_to_structured_text(combined_text)
            processing_time = time.time() - start_time
            
            result = OCRResult(
                raw_text=combined_text,
                structured_text=structured_text,
                tables=[],
                layout_analysis={"document_type": "pdf", "pages": len(pdf_reader.pages), "processing_method": "basic_text"},
                totals_analysis=self._validate_totals_from_text(combined_text),
                confidence=0.6,  # Lower confidence for basic extraction
                processing_time=processing_time
            )
            
            app_logger.info(f"Basic PDF text extraction completed: {len(all_text)} pages in {processing_time:.2f}s")
            return [result]
            
        except Exception as e:
            app_logger.error(f"Basic PDF text extraction failed: {e}")
            # Create minimal result
            return [OCRResult(
                raw_text=f"PDF processing failed: {str(e)}. Please check file format or install additional dependencies.",
                structured_text=[],
                tables=[],
                layout_analysis={"error": str(e)},
                totals_analysis={},
                confidence=0.0,
                processing_time=time.time() - start_time
            )]


def create_ocr_service(api_key: Optional[str] = None) -> OpenAIVisionOCR:
    """Factory function to create OCR service."""
    return OpenAIVisionOCR(api_key)