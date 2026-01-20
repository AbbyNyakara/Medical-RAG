"""
Gradio interface for Medical RAG Pipeline
"""

import gradio as gr
import requests
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# RAG API endpoint
API_URL = "http://localhost:8000"

# ============ API HELPER FUNCTIONS ============

def upload_document(file) -> Tuple[str, str]:
    """Upload document to API"""
    if file is None:
        return "Error", "Please select a file"
    try:
        with open(file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                message = (
                    f"✅ Document Uploaded!\n\n"
                    f"Filename: {data['filename']}\n"
                    f"Chunks: {data['total_chunks']}\n"
                    f"Vectors Stored: {data['vectors_stored']}\n"
                    f"Document ID: {data['document_id']}"
                )
                return "✅ Success", message
            else:
                return "❌ Error", f"Upload failed: {data['error']}"
        else:
            return "❌ Error", f"Server error: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return "❌ Error", f"Upload failed: {str(e)}"


def ask_question(question: str) -> Tuple[str, str]:
    """Ask question to API"""
    if not question.strip():
        return "❌ Error", "Please enter a question"
    
    try:
        payload = {
            "question": question,
            "top_k": 5
        }
        
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                answer_text = (
                    f"**Answer:**\n{data['answer']}\n\n"
                    f"**Metadata:**\n"
                    f"- Sources used: {data['num_sources']}\n"
                    f"- Processing time: {data['processing_time_seconds']:.2f}s"
                )
                return "✅ Success", answer_text
            else:
                return "❌ Error", f"Query failed: {data['error']}"
        else:
            return "❌ Error", f"Server error: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return "❌ Error", f"Query failed: {str(e)}"


# ============ GRADIO INTERFACE ============

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Medical RAG", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🏥 Medical RAG System
        
        Upload medical documents and ask questions about them.
        """)
        
        # ===== TAB 1: UPLOAD =====
        with gr.Tab("📤 Upload Document"):
            gr.Markdown("### Upload a PDF document for indexing")
            
            with gr.Column():
                file_input = gr.File(
                    label="Select PDF File",
                    file_types=[".pdf"],
                    file_count="single"
                )
                
                upload_btn = gr.Button("Upload & Index", variant="primary", size="lg")
            
            with gr.Column():
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                upload_info = gr.Textbox(
                    label="Details",
                    interactive=False,
                    lines=6
                )
            
            # Connect upload button
            upload_btn.click(
                fn=upload_document,
                inputs=[file_input],
                outputs=[upload_status, upload_info]
            )
        
        # ===== TAB 2: QUERY =====
        with gr.Tab("❓ Ask Question"):
            gr.Markdown("### Ask questions about your indexed documents")
            
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is diabetes?",
                    lines=3
                )
                
                query_btn = gr.Button("Get Answer", variant="primary", size="lg")
            
            with gr.Column():
                query_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
                
                answer_output = gr.Markdown(
                    label="Answer",
                    value="*Ask a question to get started*"
                )
            
            # Connect query button
            query_btn.click(
                fn=ask_question,
                inputs=[question_input],
                outputs=[query_status, answer_output]
            )
        
        # ===== TAB 3: INFO =====
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## Medical RAG Pipeline
            
            This system helps you:
            1. **Upload** medical documents (PDFs)
            2. **Extract** text and chunk into segments
            3. **Embed** chunks into vectors
            4. **Store** in vector database (Pinecone)
            5. **Query** and get relevant answers
            
            ### Features:
            - 🔍 Semantic search with reranking
            - 🧠 GPT-4 Turbo powered answers
            - 📊 Context-aware responses
            - ⚡ Fast retrieval
            
            ### Supported Formats:
            - PDF documents
            - Images (JPG, PNG, TIFF, WebP, BMP)
            
            ### How to use:
            1. Go to **Upload Document** tab
            2. Select a PDF file
            3. Click "Upload & Index"
            4. Go to **Ask Question** tab
            5. Type your question
            6. Get instant answers!
            """)
    
    return demo


# ============ RUN ============

if __name__ == "__main__":
    # Create interface
    interface = create_interface()
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
