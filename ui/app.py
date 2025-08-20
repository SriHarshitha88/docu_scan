import streamlit as st
import sys
from pathlib import Path

# Add project root and src directory to Python path (robust for different run contexts)
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
for p in [str(project_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.append(p)

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
                    
                    # Parse user fields
                    user_fields = None
                    if fields_to_extract.strip():
                        user_fields = [field.strip() for field in fields_to_extract.split(',')]
                    
                    # Process file with OCR and classification
                    import time
                    time.sleep(0.1)  # Small delay for UI responsiveness
                    
                    try:
                        # Read file bytes
                        file_bytes = uploaded_file.read()
                        file_type = uploaded_file.type
                        
                        # Process with OCR if available
                        result = doc_processor.process_file(
                            file_bytes, file_type, uploaded_file.name, user_fields
                        )
                        processing_results.append((uploaded_file.name, result, file_bytes, file_type))
                        
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
                    
                    for file_name, result, file_bytes, file_type in processing_results:
                        with st.expander(f"📄 {file_name} - {result.classification.document_type.value.title()}", expanded=True):
                            
                            # Classification details
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Document Type", result.classification.document_type.value.title())
                            
                            with col2:
                                st.metric("Confidence", f"{result.classification.confidence:.1%}", 
                                         delta=result.confidence_level)
                            
                            with col3:
                                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                            
                            with col4:
                                ocr_status = "🟢 OCR" if result.ocr_result else "🔵 Text"
                                st.metric("Method", ocr_status)
                            
                            # OCR Analysis (if available)
                            if result.ocr_result:
                                st.subheader("🔍 OCR Analysis")
                                
                                ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
                                with ocr_col1:
                                    st.metric("Characters", len(result.ocr_result.raw_text))
                                with ocr_col2:
                                    st.metric("Tables Found", result.table_count)
                                with ocr_col3:
                                    st.metric("OCR Confidence", f"{result.ocr_result.confidence:.1%}")
                                
                                # Table visualization
                                if result.ocr_result.tables:
                                    st.subheader("📊 Extracted Tables")
                                    
                                    # Get document processor for visualization
                                    from visualization import create_visualizer
                                    visualizer = create_visualizer()
                                    
                                    try:
                                        table_dfs = visualizer.create_table_visualization(result.ocr_result.tables)
                                        
                                        for idx, df in enumerate(table_dfs):
                                            st.write(f"**Table {idx + 1}** ({df.attrs.get('table_type', 'data')} - {df.attrs.get('confidence', 0):.1%} confidence)")
                                            st.dataframe(df, use_container_width=True)
                                    
                                    except Exception as e:
                                        st.warning(f"Table visualization failed: {e}")
                                
                                # Totals Analysis
                                if result.ocr_result.totals_analysis:
                                    st.subheader("💰 Financial Analysis")
                                    
                                    try:
                                        totals_fig = visualizer.create_totals_validation_chart(result.ocr_result.totals_analysis)
                                        if totals_fig:
                                            st.plotly_chart(totals_fig, use_container_width=True)
                                        else:
                                            st.info("No financial data detected for visualization")
                                    except Exception as e:
                                        st.warning(f"Totals visualization failed: {e}")
                                
                                # Bounding Box Visualization (for images)
                                if file_type.startswith("image/"):
                                    st.subheader("📐 Bounding Box Visualization")
                                    
                                    try:
                                        from PIL import Image
                                        import io
                                        
                                        # Load original image
                                        original_image = Image.open(io.BytesIO(file_bytes))
                                        
                                        # Create bounding box visualization
                                        bbox_image = visualizer.create_bounding_box_visualization(
                                            original_image, result.ocr_result, show_confidence=True
                                        )
                                        
                                        # Display side by side
                                        viz_col1, viz_col2 = st.columns(2)
                                        
                                        with viz_col1:
                                            st.write("**Original Image**")
                                            st.image(original_image, use_column_width=True)
                                        
                                        with viz_col2:
                                            st.write("**OCR Analysis**")
                                            st.image(bbox_image, use_column_width=True)
                                    
                                    except Exception as e:
                                        st.warning(f"Image visualization failed: {e}")
                                
                                # Layout Analysis
                                try:
                                    layout_fig = visualizer.create_layout_analysis_plot(result.ocr_result)
                                    st.subheader("📋 Layout Analysis")
                                    st.plotly_chart(layout_fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Layout visualization failed: {e}")
                            
                            # Extracted fields
                            if result.extracted_fields:
                                st.subheader("🎯 Extracted Fields")
                                
                                # Display in columns for better layout
                                field_cols = st.columns(2)
                                field_items = list(result.extracted_fields.items())
                                
                                for i, (field, value) in enumerate(field_items):
                                    if field == "__agent_meta__":
                                        continue
                                    with field_cols[i % 2]:
                                        st.write(f"**{field}:** {value}")

                                # Agent metadata and confidence bars
                                agent_meta = result.extracted_fields.get("__agent_meta__")
                                if agent_meta:
                                    st.markdown("---")
                                    st.subheader("🔎 Confidence & QA")
                                    st.write(f"**Overall Confidence:** {agent_meta.get('overall_confidence', 0):.1%}")
                                    qa = agent_meta.get("qa", {})
                                    if qa:
                                        passed = qa.get("passed_rules", [])
                                        failed = qa.get("failed_rules", [])
                                        notes = qa.get("notes")
                                        if passed:
                                            st.success("Passed rules: " + ", ".join(passed))
                                        if failed:
                                            st.warning("Failed rules: " + ", ".join(failed))
                                        if notes:
                                            st.info(notes)

                                    # Per-field confidence bars if available
                                    agent_fields = result.extracted_fields.get("__agent_fields__")
                                    if agent_fields:
                                        st.subheader("📈 Field Confidences")
                                        for f in agent_fields:
                                            name = f.get("name")
                                            conf = float(f.get("confidence") or 0.0)
                                            st.progress(conf)
                                            st.caption(f"{name}: {conf:.1%}")

                                    # JSON output (copy/download)
                                    try:
                                        json_output = {
                                            "doc_type": agent_meta.get("doc_type", "unknown"),
                                            "fields": agent_fields or [],
                                            "overall_confidence": agent_meta.get("overall_confidence", 0.0),
                                            "qa": agent_meta.get("qa", {})
                                        }
                                        st.subheader("🧾 JSON Output")
                                        st.json(json_output)
                                        import json as _json
                                        st.download_button(
                                            label="Download JSON",
                                            data=_json.dumps(json_output, indent=2).encode("utf-8"),
                                            file_name=f"{file_name}_extraction.json",
                                            mime="application/json"
                                        )
                                    except Exception as e:
                                        st.warning(f"JSON rendering failed: {e}")
                            
                            # Classification indicators
                            if result.classification.primary_indicators:
                                st.subheader("🔍 Classification Indicators")
                                st.write("**Primary:** " + ", ".join(result.classification.primary_indicators))
                                if result.classification.secondary_indicators:
                                    st.write("**Secondary:** " + ", ".join(result.classification.secondary_indicators))
                            
                            # Suggested fields for this document type
                            st.subheader("💡 Suggested Fields for This Document Type")
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