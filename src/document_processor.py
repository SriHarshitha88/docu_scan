from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time

from document_classifier import create_classifier, DocumentType, ClassificationResult
from ml_fallback import create_hybrid_classifier
from ocr_service import create_ocr_service, OCRResult
from visualization import create_visualizer
from extraction_agent import create_extraction_agent, ExtractionOutput, AgentConfig
from logger import app_logger


@dataclass
class ProcessingResult:
    """Result of document processing including classification and extraction."""
    classification: ClassificationResult
    extracted_fields: Dict[str, Any]
    processing_time: float
    suggested_fields: List[str]
    confidence_level: str
    ocr_result: Optional[OCRResult] = None
    has_tables: bool = False
    table_count: int = 0
    extraction_method: str = "text"  # "text", "ocr", "hybrid"


class DocumentProcessor:
    """
    Main document processor that handles classification and field extraction.
    """
    
    def __init__(self):
        """Initialize document processor with classifiers and OCR."""
        app_logger.info("Initializing document processor")
        
        # Initialize heuristic classifier
        self.heuristic_classifier = create_classifier()
        
        # Initialize hybrid classifier with ML fallback
        self.hybrid_classifier = create_hybrid_classifier(self.heuristic_classifier)
        
        # Initialize OCR service (will be None if no API key)
        try:
            self.ocr_service = create_ocr_service()
            app_logger.info("OCR service initialized successfully")
        except Exception as e:
            app_logger.warning(f"OCR service not available: {e}")
            self.ocr_service = None
        
        # Initialize visualizer
        self.visualizer = create_visualizer()
        
        # Field extraction templates based on document types
        self._setup_extraction_templates()

        # Extraction agent
        try:
            self.extraction_agent = create_extraction_agent()
            app_logger.info("Extraction agent initialized")
        except Exception as e:
            app_logger.warning(f"Extraction agent not available: {e}")
            self.extraction_agent = None
    
    def _setup_extraction_templates(self):
        """Setup field extraction templates for different document types."""
        
        self.extraction_templates = {
            DocumentType.INVOICE: [
                "Invoice Number",
                "Invoice Date", 
                "Due Date",
                "Vendor Name",
                "Bill To",
                "Total Amount",
                "Tax Amount",
                "Subtotal",
                "Payment Terms",
                "Description",
                "Quantity",
                "Unit Price"
            ],
            
            DocumentType.BILL: [
                "Account Number",
                "Billing Period",
                "Due Date",
                "Amount Due",
                "Previous Balance",
                "Current Charges",
                "Service Address",
                "Billing Address",
                "Usage Amount",
                "Service Type"
            ],
            
            DocumentType.RECEIPT: [
                "Store Name",
                "Transaction Date",
                "Transaction Number",
                "Total Amount",
                "Tax Amount",
                "Payment Method",
                "Cashier ID",
                "Items Purchased",
                "Change Given"
            ],
            
            DocumentType.CONTRACT: [
                "Contract Number",
                "Effective Date",
                "Expiration Date",
                "Party 1",
                "Party 2",
                "Contract Value",
                "Terms",
                "Signature Date",
                "Witness"
            ],
            
            DocumentType.MEDICAL: [
                "Patient Name",
                "Date of Birth",
                "Doctor Name",
                "Visit Date",
                "Diagnosis",
                "Medications",
                "Dosage",
                "Instructions",
                "Insurance Information",
                "Medical Record Number"
            ],
            
            DocumentType.FINANCIAL: [
                "Account Number",
                "Statement Date",
                "Account Holder",
                "Balance",
                "Transactions",
                "Interest Rate",
                "Fees",
                "Bank Name",
                "Routing Number"
            ],
            
            DocumentType.LEGAL: [
                "Case Number",
                "Court Name",
                "Plaintiff",
                "Defendant",
                "Attorney",
                "Filing Date",
                "Hearing Date",
                "Judgment",
                "Legal Citations"
            ]
        }
    
    def process_file(self, file_data: bytes, file_type: str, file_name: str,
                    user_fields: Optional[List[str]] = None) -> ProcessingResult:
        """
        Process document file (PDF or image) with OCR and classification.
        
        Args:
            file_data: File bytes
            file_type: File MIME type
            file_name: Original file name
            user_fields: Optional user-specified fields to extract
            
        Returns:
            ProcessingResult with comprehensive analysis
        """
        start_time = time.time()
        app_logger.info(f"Processing file: {file_name} ({file_type})")
        
        ocr_result = None
        extraction_method = "text"
        
        # Determine processing method based on file type
        if file_type == "application/pdf":
            text, ocr_result = self._process_pdf(file_data)
            extraction_method = "ocr" if ocr_result else "text"
        elif file_type.startswith("image/"):
            text, ocr_result = self._process_image(file_data)  
            extraction_method = "ocr" if ocr_result else "text"
        else:
            # Text-based processing fallback
            text = f"Unsupported file type: {file_type}"
            app_logger.warning(f"Unsupported file type: {file_type}")
        
        # Continue with standard document processing, pass OCR if available
        result = self.process_document(text, user_fields, ocr_result)
        
        # Enhance result with OCR data
        result.ocr_result = ocr_result
        result.extraction_method = extraction_method
        
        if ocr_result:
            result.has_tables = len(ocr_result.tables) > 0
            result.table_count = len(ocr_result.tables)
            
            # Enhanced field extraction using OCR data
            result.extracted_fields.update(
                self._extract_fields_from_ocr(ocr_result, result.suggested_fields)
            )
        
        app_logger.info(f"File processing completed in {result.processing_time:.2f}s")
        return result
    
    def process_document(self, text: str, user_fields: Optional[List[str]] = None,
                        ocr: Optional[OCRResult] = None) -> ProcessingResult:
        """
        Process document with classification and field extraction.
        
        Args:
            text: Document text content
            user_fields: Optional user-specified fields to extract
            
        Returns:
            ProcessingResult with classification and extraction results
        """
        start_time = time.time()
        
        app_logger.info("Starting document processing")
        
        # Step 1: Classify document
        classification_result = self.hybrid_classifier.classify_document(text)
        
        # Step 2: Get suggested fields based on classification
        suggested_fields = self._get_suggested_fields(
            classification_result.document_type, user_fields
        )
        
        # Step 3: Extract fields using agent if available, otherwise fallback
        extracted_fields = {}
        if self.extraction_agent:
            try:
                agent_out: ExtractionOutput = self.extraction_agent.extract(
                    text=text,
                    classification=classification_result,
                    ocr=ocr,
                    user_fields=suggested_fields,
                    config=AgentConfig(num_samples=3, temperature=0.3)
                )

                # Convert to flat dict for backward compatibility
                extracted_fields = {f.name: f.value for f in agent_out.fields}
                # Attach QA-style info into extracted fields under a special key
                extracted_fields["__agent_meta__"] = {
                    "doc_type": agent_out.doc_type,
                    "overall_confidence": agent_out.overall_confidence,
                    "qa": agent_out.qa.dict(),
                }
                # Attach detailed fields for UI
                extracted_fields["__agent_fields__"] = [
                    {"name": f.name, "value": f.value, "confidence": f.confidence,
                     "source": f.source.dict() if f.source else None}
                    for f in agent_out.fields
                ]
            except Exception as e:
                app_logger.warning(f"Agent extraction failed, using heuristic extraction: {e}")
                extracted_fields = self._extract_fields(text, suggested_fields, classification_result)
        else:
            extracted_fields = self._extract_fields(text, suggested_fields, classification_result)
        
        # Step 4: Calculate processing time
        processing_time = time.time() - start_time
        
        # Step 5: Determine confidence level
        confidence_level = self._get_confidence_level(classification_result.confidence)
        
        result = ProcessingResult(
            classification=classification_result,
            extracted_fields=extracted_fields,
            processing_time=processing_time,
            suggested_fields=suggested_fields,
            confidence_level=confidence_level
        )
        
        app_logger.info(f"Document processing completed in {processing_time:.2f}s")
        
        return result
    
    def _get_suggested_fields(self, document_type: DocumentType, 
                            user_fields: Optional[List[str]] = None) -> List[str]:
        """Get suggested fields for extraction based on document type."""
        
        # Get template fields for document type
        template_fields = self.extraction_templates.get(document_type, [])
        
        if user_fields:
            # Combine user fields with template suggestions
            user_field_list = [field.strip() for field in user_fields if field.strip()]
            
            # Prioritize user fields, then add template suggestions
            suggested = user_field_list.copy()
            
            for field in template_fields:
                if field not in suggested:
                    suggested.append(field)
            
            return suggested[:15]  # Limit to top 15 fields
        
        return template_fields[:10]  # Default to top 10 template fields
    
    def _extract_fields(self, text: str, fields: List[str], 
                       classification: ClassificationResult) -> Dict[str, Any]:
        """
        Extract specified fields from document text.
        This is a placeholder implementation - would be enhanced with actual extraction logic.
        """
        
        # Placeholder extraction - would implement actual field extraction here
        extracted = {}
        
        # Mock some extractions based on classification confidence
        if classification.confidence > 0.8:
            # High confidence - extract more fields
            for i, field in enumerate(fields[:6]):
                extracted[field] = f"[Extracted from {classification.document_type.value}]"
        elif classification.confidence > 0.5:
            # Medium confidence - extract fewer fields
            for i, field in enumerate(fields[:3]):
                extracted[field] = f"[Extracted from {classification.document_type.value}]"
        else:
            # Low confidence - minimal extraction
            if fields:
                extracted[fields[0]] = f"[Low confidence extraction]"
        
        return extracted
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to descriptive level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def get_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return [doc_type.value for doc_type in DocumentType if doc_type != DocumentType.UNKNOWN]
    
    def get_template_fields(self, document_type: str) -> List[str]:
        """Get template fields for a specific document type."""
        try:
            doc_type_enum = DocumentType(document_type)
            return self.extraction_templates.get(doc_type_enum, [])
        except ValueError:
            return []
    
    def _process_pdf(self, pdf_data: bytes) -> tuple[str, Optional[OCRResult]]:
        """Process PDF file using OpenAI API directly."""
        text = ""
        ocr_result = None
        
        try:
            if self.ocr_service:
                # Use OpenAI for direct PDF processing
                app_logger.info("Processing PDF with OpenAI API")
                ocr_results = self.ocr_service.extract_from_pdf(pdf_data)
                
                if ocr_results:
                    # Combine text from all pages/results
                    text = "\n\n--- PAGE BREAK ---\n\n".join(
                        result.raw_text for result in ocr_results
                    )
                    
                    # Use first result for detailed analysis
                    ocr_result = ocr_results[0]
                    
                    app_logger.info(f"PDF processing completed: {len(ocr_results)} results")
                    return text, ocr_result
                else:
                    text = "PDF processing failed - no content extracted"
            else:
                # No OCR service available
                app_logger.warning("PDF processing requires OpenAI API key")
                text = "PDF processing requires OpenAI API key for full functionality"
                
        except Exception as e:
            app_logger.error(f"PDF processing failed: {e}")
            text = f"PDF processing error: {str(e)}"
        
        return text, ocr_result
    
    def _process_image(self, image_data: bytes) -> tuple[str, Optional[OCRResult]]:
        """Process image file with OCR if available."""
        text = ""
        ocr_result = None
        
        try:
            if self.ocr_service:
                # Use OCR for image processing
                app_logger.info("Processing image with OCR")
                ocr_result = self.ocr_service.extract_from_image(image_data)
                text = ocr_result.raw_text
                
                app_logger.info(f"Image OCR completed: {len(text)} characters extracted")
            else:
                # No OCR available
                app_logger.warning("Image processing requires OCR service")
                text = "Image processing requires OCR service - please configure OpenAI API key"
                
        except Exception as e:
            app_logger.error(f"Image processing failed: {e}")
            text = f"Image processing error: {str(e)}"
        
        return text, ocr_result
    
    def _extract_fields_from_ocr(self, ocr_result: OCRResult, 
                                suggested_fields: List[str]) -> Dict[str, Any]:
        """Extract specific fields using OCR data and table information."""
        extracted = {}
        
        try:
            # Extract from tables if available
            if ocr_result.tables:
                table_data = self._extract_from_tables(ocr_result.tables, suggested_fields)
                extracted.update(table_data)
            
            # Extract from totals analysis
            if ocr_result.totals_analysis:
                totals_data = self._extract_from_totals(ocr_result.totals_analysis, suggested_fields)
                extracted.update(totals_data)
            
            # Extract from structured text
            text_data = self._extract_from_structured_text(ocr_result.structured_text, suggested_fields)
            extracted.update(text_data)
            
            app_logger.info(f"OCR field extraction completed: {len(extracted)} fields found")
            
        except Exception as e:
            app_logger.warning(f"OCR field extraction failed: {e}")
        
        return extracted
    
    def _extract_from_tables(self, tables: List, suggested_fields: List[str]) -> Dict[str, Any]:
        """Extract fields from table data."""
        extracted = {}
        
        for table_idx, table in enumerate(tables):
            # Extract totals from tables
            for cell in table.cells:
                cell_content = cell.content.lower()
                
                # Look for total amounts
                if any(word in cell_content for word in ['total', 'amount', 'sum']):
                    # Try to extract numeric value
                    import re
                    amounts = re.findall(r'\$?\d+(?:\.\d+)?', cell.content)
                    if amounts:
                        key = f"Table_{table_idx + 1}_Total"
                        extracted[key] = amounts[0]
                
                # Look for dates
                if any(word in cell_content for word in ['date', 'due']):
                    dates = re.findall(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', cell.content)
                    if dates:
                        key = f"Table_{table_idx + 1}_Date"
                        extracted[key] = dates[0]
        
        return extracted
    
    def _extract_from_totals(self, totals_analysis: Dict[str, Any], 
                           suggested_fields: List[str]) -> Dict[str, Any]:
        """Extract fields from totals analysis."""
        extracted = {}
        
        amounts = totals_analysis.get('amounts', [])
        for amount in amounts:
            context = amount.get('context', 'amount')
            value = amount.get('value', 0)
            
            # Map contexts to field names
            field_mapping = {
                'total': 'Total Amount',
                'subtotal': 'Subtotal',
                'tax': 'Tax Amount',
                'discount': 'Discount',
                'balance': 'Balance Due'
            }
            
            field_name = field_mapping.get(context, context.title())
            extracted[field_name] = f"${value:.2f}"
        
        return extracted
    
    def _extract_from_structured_text(self, structured_text: List, 
                                    suggested_fields: List[str]) -> Dict[str, Any]:
        """Extract fields from structured text elements."""
        extracted = {}
        
        for text_element in structured_text:
            element_text = text_element.text
            element_type = text_element.element_type
            
            # Extract based on element type and content
            if element_type == 'date':
                extracted['Document Date'] = element_text
            elif element_type == 'currency' and 'Total' not in extracted:
                extracted['Amount'] = element_text
            elif element_type == 'header' and len(element_text) > 5:
                extracted['Document Title'] = element_text
        
        return extracted


def create_document_processor() -> DocumentProcessor:
    """Factory function to create a document processor."""
    return DocumentProcessor()