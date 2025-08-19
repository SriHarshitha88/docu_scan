"""
Visualization system for OCR results including bounding boxes, tables, and layout analysis.
"""

import io
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ocr_service import OCRResult, ExtractedTable, ExtractedText, BoundingBox
from logger import app_logger


class OCRVisualizer:
    """
    Visualization tools for OCR results including bounding boxes, tables, and confidence maps.
    """
    
    def __init__(self):
        """Initialize visualizer with color schemes and fonts."""
        self.color_schemes = {
            "text": "#4CAF50",      # Green
            "table_cell": "#2196F3", # Blue  
            "header": "#FF9800",     # Orange
            "footer": "#9C27B0",     # Purple
            "financial": "#F44336",  # Red
            "currency": "#FF5722",   # Deep Orange
            "date": "#607D8B"        # Blue Grey
        }
        
        self.table_colors = [
            "#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", 
            "#42A5F5", "#2196F3", "#1E88E5", "#1976D2"
        ]
        
    def create_bounding_box_visualization(self, image: Image.Image, 
                                        ocr_result: OCRResult,
                                        show_confidence: bool = True) -> Image.Image:
        """
        Create visualization with bounding boxes overlaid on the original image.
        
        Args:
            image: Original document image
            ocr_result: OCR extraction results
            show_confidence: Whether to show confidence scores
            
        Returns:
            PIL Image with bounding boxes drawn
        """
        app_logger.info("Creating bounding box visualization")
        
        # Create a copy of the image
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font (fall back to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw text element bounding boxes
        for text_element in ocr_result.structured_text:
            self._draw_text_bbox(draw, text_element, font, show_confidence)
        
        # Draw table bounding boxes
        for table in ocr_result.tables:
            self._draw_table_bbox(draw, table, font)
        
        app_logger.info("Bounding box visualization completed")
        return vis_image
    
    def create_table_visualization(self, tables: List[ExtractedTable]) -> List[pd.DataFrame]:
        """
        Create pandas DataFrames for table visualization.
        
        Args:
            tables: List of extracted tables
            
        Returns:
            List of DataFrames for each table
        """
        app_logger.info(f"Creating table visualizations for {len(tables)} tables")
        
        dataframes = []
        
        for table_idx, table in enumerate(tables):
            try:
                # Create matrix for table data
                if table.rows == 0 or table.columns == 0:
                    continue
                    
                data_matrix = [["" for _ in range(table.columns)] for _ in range(table.rows)]
                headers = None
                
                # Fill matrix with cell data
                for cell in table.cells:
                    if 0 <= cell.row < table.rows and 0 <= cell.column < table.columns:
                        data_matrix[cell.row][cell.column] = cell.content
                        
                        # Collect headers
                        if cell.is_header:
                            if headers is None:
                                headers = [""] * table.columns
                            if cell.column < len(headers):
                                headers[cell.column] = cell.content
                
                # Create DataFrame
                if headers and any(h.strip() for h in headers):
                    # Remove header row from data if present
                    data_rows = data_matrix[1:] if len(data_matrix) > 1 else data_matrix
                    df = pd.DataFrame(data_rows, columns=headers)
                else:
                    df = pd.DataFrame(data_matrix)
                
                # Add metadata
                df.attrs['table_id'] = table_idx
                df.attrs['table_type'] = table.table_type
                df.attrs['confidence'] = table.confidence
                df.attrs['position'] = {
                    'x': table.bbox.x,
                    'y': table.bbox.y,
                    'width': table.bbox.width,
                    'height': table.bbox.height
                }
                
                dataframes.append(df)
                
            except Exception as e:
                app_logger.warning(f"Table visualization failed for table {table_idx}: {e}")
                continue
        
        app_logger.info(f"Created {len(dataframes)} table visualizations")
        return dataframes
    
    def create_confidence_heatmap(self, ocr_result: OCRResult, 
                                image_width: int, image_height: int) -> go.Figure:
        """
        Create confidence heatmap showing extraction quality across the document.
        
        Args:
            ocr_result: OCR extraction results
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Plotly heatmap figure
        """
        app_logger.info("Creating confidence heatmap")
        
        # Create grid for confidence mapping
        grid_size = 20
        x_bins = max(1, image_width // grid_size)
        y_bins = max(1, image_height // grid_size)
        
        confidence_grid = np.zeros((y_bins, x_bins))
        count_grid = np.zeros((y_bins, x_bins))
        
        # Map text elements to grid
        for text_element in ocr_result.structured_text:
            x_idx = min(int(text_element.bbox.x / grid_size), x_bins - 1)
            y_idx = min(int(text_element.bbox.y / grid_size), y_bins - 1)
            
            confidence_grid[y_idx, x_idx] += text_element.confidence
            count_grid[y_idx, x_idx] += 1
        
        # Average confidence in each grid cell
        with np.errstate(divide='ignore', invalid='ignore'):
            confidence_grid = np.divide(confidence_grid, count_grid, 
                                      out=np.zeros_like(confidence_grid), 
                                      where=count_grid!=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confidence_grid,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Confidence")
        ))
        
        fig.update_layout(
            title="OCR Confidence Heatmap",
            xaxis_title="X Position (grid)",
            yaxis_title="Y Position (grid)",
            width=800,
            height=600
        )
        
        return fig
    
    def create_layout_analysis_plot(self, ocr_result: OCRResult) -> go.Figure:
        """
        Create layout analysis visualization showing document structure.
        
        Args:
            ocr_result: OCR extraction results
            
        Returns:
            Plotly scatter plot figure
        """
        app_logger.info("Creating layout analysis plot")
        
        # Prepare data for plotting
        x_coords = []
        y_coords = []
        texts = []
        colors = []
        sizes = []
        
        for text_element in ocr_result.structured_text:
            x_coords.append(text_element.bbox.x + text_element.bbox.width / 2)
            y_coords.append(text_element.bbox.y + text_element.bbox.height / 2)
            texts.append(text_element.text[:30] + "..." if len(text_element.text) > 30 else text_element.text)
            colors.append(self.color_schemes.get(text_element.element_type, "#808080"))
            sizes.append(min(text_element.confidence * 20, 15))
        
        # Create scatter plot
        fig = go.Figure()
        
        # Group by element type for legend
        element_types = set(text.element_type for text in ocr_result.structured_text)
        
        for element_type in element_types:
            type_indices = [i for i, text in enumerate(ocr_result.structured_text) 
                          if text.element_type == element_type]
            
            if type_indices:
                fig.add_trace(go.Scatter(
                    x=[x_coords[i] for i in type_indices],
                    y=[y_coords[i] for i in type_indices],
                    mode='markers+text',
                    name=element_type.title(),
                    marker=dict(
                        color=self.color_schemes.get(element_type, "#808080"),
                        size=[sizes[i] for i in type_indices],
                        opacity=0.7
                    ),
                    text=[texts[i] for i in type_indices],
                    textposition="top center",
                    textfont=dict(size=8)
                ))
        
        # Add table rectangles
        for table_idx, table in enumerate(ocr_result.tables):
            fig.add_shape(
                type="rect",
                x0=table.bbox.x,
                y0=table.bbox.y,
                x1=table.bbox.x2,
                y1=table.bbox.y2,
                line=dict(color="red", width=2),
                fillcolor="rgba(255,0,0,0.1)"
            )
            
            # Add table label
            fig.add_annotation(
                x=table.bbox.x + table.bbox.width / 2,
                y=table.bbox.y,
                text=f"Table {table_idx + 1}",
                showarrow=False,
                bgcolor="red",
                bordercolor="white",
                font=dict(color="white", size=10)
            )
        
        fig.update_layout(
            title="Document Layout Analysis",
            xaxis_title="X Position",
            yaxis_title="Y Position", 
            yaxis=dict(autorange="reversed"),  # Flip Y axis to match image coordinates
            width=1000,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_totals_validation_chart(self, totals_analysis: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create visualization for financial totals validation.
        
        Args:
            totals_analysis: Totals analysis from OCR
            
        Returns:
            Plotly bar chart or None if no data
        """
        if not totals_analysis.get('amounts'):
            return None
        
        app_logger.info("Creating totals validation chart")
        
        amounts = totals_analysis['amounts']
        
        # Prepare data
        contexts = [amount['context'] for amount in amounts]
        values = [amount['value'] for amount in amounts]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=contexts,
                y=values,
                marker_color=['green' if context != 'total' else 'blue' for context in contexts],
                text=[f"${value:.2f}" for value in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Financial Amounts Analysis",
            xaxis_title="Amount Type",
            yaxis_title="Value ($)",
            width=600,
            height=400
        )
        
        # Add validation status
        calculations = totals_analysis.get('calculations', {})
        validation_text = []
        
        if calculations.get('subtotal_matches'):
            validation_text.append("✅ Subtotal valid")
        if calculations.get('tax_calculation_valid'):
            validation_text.append("✅ Tax calculation valid") 
        if calculations.get('total_matches'):
            validation_text.append("✅ Total matches")
        
        if calculations.get('discrepancies'):
            validation_text.extend([f"❌ {disc}" for disc in calculations['discrepancies']])
        
        if validation_text:
            fig.add_annotation(
                text="<br>".join(validation_text),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        
        return fig
    
    def _draw_text_bbox(self, draw: ImageDraw.Draw, text_element: ExtractedText, 
                       font, show_confidence: bool = True):
        """Draw bounding box for text element."""
        bbox = text_element.bbox
        color = self.color_schemes.get(text_element.element_type, "#808080")
        
        # Draw rectangle
        draw.rectangle(
            [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
            outline=color,
            width=2
        )
        
        # Add label with confidence if requested
        if show_confidence:
            label = f"{text_element.element_type}: {text_element.confidence:.1%}"
            draw.text(
                (bbox.x, bbox.y - 15),
                label,
                fill=color,
                font=font
            )
    
    def _draw_table_bbox(self, draw: ImageDraw.Draw, table: ExtractedTable, font):
        """Draw bounding box for table."""
        bbox = table.bbox
        
        # Draw table outline
        draw.rectangle(
            [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
            outline="red",
            width=3
        )
        
        # Add table label
        label = f"Table ({table.rows}x{table.columns})"
        draw.text(
            (bbox.x, bbox.y - 20),
            label,
            fill="red",
            font=font
        )
        
        # Draw cell boundaries (simplified)
        if table.cells:
            for cell in table.cells:
                cell_bbox = cell.bbox
                color = "blue" if cell.is_header else "gray"
                draw.rectangle(
                    [(cell_bbox.x, cell_bbox.y), (cell_bbox.x2, cell_bbox.y2)],
                    outline=color,
                    width=1
                )


def create_visualizer() -> OCRVisualizer:
    """Factory function to create visualizer."""
    return OCRVisualizer()