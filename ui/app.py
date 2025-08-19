import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import settings
from logger import app_logger
from document_processor import create_document_processor


def get_ui_config():
    """Get constant UI configuration settings."""
    return {
        "layout": "wide",
        "show_emojis": True,
        "compact_mode": True,
        "show_file_details": True,
        "auto_expand_sections": True
    }


@st.cache_resource
def get_document_processor():
    """Get cached document processor instance."""
    return create_document_processor()


def main():
    """Main DocuScan application."""
    
    # Page configuration
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Get UI configuration and document processor
    ui_config = get_ui_config()
    doc_processor = get_document_processor()
    
    # Apply wide layout configuration
    main_col = st.container()
    
    with main_col:
        # Header with emoji configuration
        if ui_config["show_emojis"]:
            st.title("📄 DocuScan")
            st.subheader("Upload your documents to begin")
        else:
            st.title("DocuScan")
            st.subheader("Upload your documents to begin")
        
        if not ui_config["compact_mode"]:
            st.markdown("---")
    
        # Document Upload Section
        emoji_prefix = "📂 " if ui_config["show_emojis"] else ""
        st.header(f"{emoji_prefix}Document Upload")
        
        if not ui_config["compact_mode"]:
            st.markdown("Upload PDF files or images (PNG, JPG, JPEG) to extract information from them.")
        
        uploaded_files = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help=f"Maximum file size: {settings.upload_max_size_mb}MB per file"
        )
        
        # Field Specification Input
        emoji_prefix = "🎯 " if ui_config["show_emojis"] else ""
        st.header(f"{emoji_prefix}Fields to Extract")
        
        if not ui_config["compact_mode"]:
            st.markdown("Optionally specify what information you want to extract from your documents.")
        
        fields_to_extract = st.text_area(
            "Fields to Extract (Optional)",
            placeholder="e.g., Invoice Number, Total Amount, Date, Company Name, Address",
            help="Enter the specific fields you want to extract, separated by commas. Leave blank to extract all available information.",
            height=60 if ui_config["compact_mode"] else 100
        )
        
        # Process Button and Status
        if not ui_config["compact_mode"]:
            st.markdown("---")
        
        emoji_prefix = "🚀 " if ui_config["show_emojis"] else ""
        process_button = st.button(
            f"{emoji_prefix}Process Documents",
            type="primary",
            disabled=not uploaded_files
        )
    
        # Output Section
        emoji_prefix = "📊 " if ui_config["show_emojis"] else ""
        st.header(f"{emoji_prefix}Status")
        
        if uploaded_files:
            emoji_prefix = "✅ " if ui_config["show_emojis"] else ""
            st.success(f"{emoji_prefix}{len(uploaded_files)} file(s) uploaded successfully!")
            
            # Log file uploads
            app_logger.info(f"Files uploaded: {[file.name for file in uploaded_files]}")
            
            # Display file details
            if ui_config["show_file_details"]:
                emoji_prefix = "📋 " if ui_config["show_emojis"] else ""
                with st.expander(f"{emoji_prefix}Uploaded File Details", expanded=ui_config["auto_expand_sections"]):
                    for file in uploaded_files:
                        file_size_mb = file.size / (1024 * 1024)
                        st.write(f"**{file.name}** - {file_size_mb:.2f} MB ({file.type})")
        
        if fields_to_extract.strip():
            emoji_prefix = "📝 " if ui_config["show_emojis"] else ""
            st.info(f"{emoji_prefix}Fields to extract: {fields_to_extract}")
            app_logger.info(f"Fields specified: {fields_to_extract}")
        else:
            if uploaded_files:
                emoji_prefix = "📝 " if ui_config["show_emojis"] else ""
                st.info(f"{emoji_prefix}No specific fields specified - will extract all available information")
        
        if process_button:
            if uploaded_files:
                # Show processing indicator
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                results_placeholder = st.empty()
                
                with progress_placeholder:
                    progress_bar = st.progress(0)
                    st.write("🔄 Processing documents...")
                
                # Process each file
                processing_results = []
                
                for file_idx, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress_bar.progress(int((file_idx / len(uploaded_files)) * 100))
                    status_placeholder.info(f"📄 Processing {uploaded_file.name}...")
                    
                    # Read file content (simulate text extraction)
                    if uploaded_file.type == "application/pdf":
                        # For PDF files, simulate text extraction
                        file_text = f"Sample text content from {uploaded_file.name}. This would contain the actual extracted text from the PDF document including invoice numbers, dates, amounts, and other relevant information."
                    else:
                        # For images, simulate OCR text extraction
                        file_text = f"OCR extracted text from {uploaded_file.name}. This would contain text recognized from the image including printed and handwritten content."
                    
                    # Parse user fields
                    user_fields = None
                    if fields_to_extract.strip():
                        user_fields = [field.strip() for field in fields_to_extract.split(',')]
                    
                    # Process document with classification and extraction
                    import time
                    time.sleep(0.5)  # Simulate processing time
                    
                    try:
                        result = doc_processor.process_document(file_text, user_fields)
                        processing_results.append((uploaded_file.name, result))
                        
                        status_placeholder.success(f"✅ {uploaded_file.name} - {result.classification.document_type.value.title()} ({result.confidence_level} confidence)")
                        
                    except Exception as e:
                        app_logger.error(f"Processing failed for {uploaded_file.name}: {e}")
                        status_placeholder.error(f"❌ Failed to process {uploaded_file.name}")
                
                # Complete progress
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Show completion message
                emoji_prefix = "🎉 " if ui_config["show_emojis"] else ""
                st.success(f"{emoji_prefix}Processing completed! Analyzed {len(processing_results)} documents")
                
                # Display detailed results
                with results_placeholder:
                    st.subheader("📊 Processing Results")
                    
                    for file_name, result in processing_results:
                        with st.expander(f"📄 {file_name} - {result.classification.document_type.value.title()}", expanded=True):
                            
                            # Classification details
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Document Type", result.classification.document_type.value.title())
                            
                            with col2:
                                st.metric("Confidence", f"{result.classification.confidence:.1%}", 
                                         delta=result.confidence_level)
                            
                            with col3:
                                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                            
                            # Extracted fields
                            if result.extracted_fields:
                                st.subheader("🎯 Extracted Fields")
                                for field, value in result.extracted_fields.items():
                                    st.write(f"**{field}:** {value}")
                            
                            # Classification indicators
                            if result.classification.primary_indicators:
                                st.subheader("🔍 Classification Indicators")
                                st.write("**Primary:** " + ", ".join(result.classification.primary_indicators))
                                if result.classification.secondary_indicators:
                                    st.write("**Secondary:** " + ", ".join(result.classification.secondary_indicators))
                            
                            # Suggested fields for this document type
                            st.subheader("💡 Suggested Fields")
                            suggested_cols = st.columns(3)
                            for i, field in enumerate(result.suggested_fields[:9]):
                                with suggested_cols[i % 3]:
                                    st.write(f"• {field}")
                
                # Log processing event
                fields_msg = fields_to_extract.strip() if fields_to_extract.strip() else "auto-detected fields"
                app_logger.info(f"Processing completed for {len(uploaded_files)} files with fields: {fields_msg}")
            else:
                emoji_prefix = "⚠️ " if ui_config["show_emojis"] else ""
                st.error(f"{emoji_prefix}Please upload files to process")
        
        # Footer
        if not ui_config["compact_mode"]:
            st.markdown("---")
            st.markdown(
                "<div style='text-align: center; color: #666;'>"
                "DocuScan v1.0 - Phase 1: Document Upload Foundation"
                "</div>",
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    try:
        app_logger.info("Starting DocuScan application")
        main()
    except Exception as e:
        app_logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        raise