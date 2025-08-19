"""
Document processing integration that combines classification with extraction.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from document_classifier import create_classifier, DocumentType, ClassificationResult
from ml_fallback import create_hybrid_classifier
from logger import app_logger


@dataclass
class ProcessingResult:
    """Result of document processing including classification and extraction."""
    classification: ClassificationResult
    extracted_fields: Dict[str, Any]
    processing_time: float
    suggested_fields: List[str]
    confidence_level: str


class DocumentProcessor:
    """
    Main document processor that handles classification and field extraction.
    """
    
    def __init__(self):
        """Initialize document processor with classifiers."""
        app_logger.info("Initializing document processor")
        
        # Initialize heuristic classifier
        self.heuristic_classifier = create_classifier()
        
        # Initialize hybrid classifier with ML fallback
        self.hybrid_classifier = create_hybrid_classifier(self.heuristic_classifier)
        
        # Field extraction templates based on document types
        self._setup_extraction_templates()
    
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
    
    def process_document(self, text: str, user_fields: Optional[List[str]] = None) -> ProcessingResult:
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
        
        # Step 3: Extract fields (placeholder for now)
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


def create_document_processor() -> DocumentProcessor:
    """Factory function to create a document processor."""
    return DocumentProcessor()