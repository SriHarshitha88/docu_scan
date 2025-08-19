import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import settings
from logger import app_logger


def main():
    """Main DocuScan application."""
    
    # Page configuration
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Header
    st.title("📄 DocuScan")
    st.subheader("Upload your documents to begin")
    st.markdown("---")
    
    # Document Upload Section
    st.header("📂 Document Upload")
    st.markdown("Upload PDF files or images (PNG, JPG, JPEG) to extract information from them.")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help=f"Maximum file size: {settings.upload_max_size_mb}MB per file"
    )
    
    # Field Specification Input
    st.header("🎯 Fields to Extract")
    st.markdown("Specify what information you want to extract from your documents.")
    
    fields_to_extract = st.text_area(
        "Fields to Extract",
        placeholder="e.g., Invoice Number, Total Amount, Date, Company Name, Address",
        help="Enter the specific fields you want to extract, separated by commas",
        height=100
    )
    
    # Process Button and Status
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files or not fields_to_extract.strip()
        )
    
    # Output Section
    st.header("📊 Status")
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Log file uploads
        app_logger.info(f"Files uploaded: {[file.name for file in uploaded_files]}")
        
        # Display file details
        with st.expander("📋 Uploaded File Details", expanded=True):
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                st.write(f"**{file.name}** - {file_size_mb:.2f} MB ({file.type})")
    
    if fields_to_extract.strip():
        st.info(f"📝 Fields to extract: {fields_to_extract}")
        app_logger.info(f"Fields specified: {fields_to_extract}")
    
    if process_button:
        if uploaded_files and fields_to_extract.strip():
            st.balloons()
            st.success("🎉 Processing completed! (Phase 1 - Upload confirmation)")
            
            # Log processing event
            app_logger.info(f"Processing initiated for {len(uploaded_files)} files with fields: {fields_to_extract}")
            
            # Display what would happen in future phases
            st.info("""
            **Next Steps (Future Phases):**
            - Document text extraction
            - AI-powered field identification
            - Structured data export
            """)
        else:
            st.error("⚠️ Please upload files and specify fields to extract")
    
    # Footer
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