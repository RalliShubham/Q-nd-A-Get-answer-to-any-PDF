import streamlit as st
import os
from pypdf import PdfReader
from transformers import pipeline
import tempfile

# Set page config
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_qa_model():
    return pipeline(
        task="question-answering", 
        model="distilbert-base-cased-distilled-squad"
    )

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Read the PDF
        reader = PdfReader(tmp_file_path)
        document_text = ""
        
        for page in reader.pages:
            document_text += page.extract_text() + "\n"
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return document_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def get_answer(question, context, qa_pipeline):
    """Get answer from the QA pipeline"""
    try:
        # Truncate context if too long
        max_context_length = 4000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        result = qa_pipeline(question=question, context=context)
        return result
    except Exception as e:
        st.error(f"Error getting answer: {str(e)}")
        return None

def main():
    st.title("📄 PDF Question Answering System")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize session state
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            if st.session_state.filename != uploaded_file.name:
                st.session_state.filename = uploaded_file.name
                with st.spinner("Extracting text from PDF..."):
                    st.session_state.document_text = extract_text_from_pdf(uploaded_file)
                
                if st.session_state.document_text:
                    st.success("✅ PDF processed successfully!")
                    st.info(f"📊 Extracted {len(st.session_state.document_text)} characters")
    
    # Main content area
    if st.session_state.document_text:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🤖 Ask Questions")
            
            # Load QA model
            with st.spinner("Loading AI model..."):
                qa_pipeline = load_qa_model()
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the notice period for resignation?",
                key="question_input"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question.strip():
                    with st.spinner("Finding answer..."):
                        result = get_answer(question, st.session_state.document_text, qa_pipeline)
                    
                    if result:
                        st.subheader("💡 Answer:")
                        st.write(result['answer'])
                        
                        confidence = round(result.get('score', 0) * 100, 1)
                        st.metric("Confidence", f"{confidence}%")
                        
                        if confidence < 50:
                            st.warning("⚠️ Low confidence answer. The information might not be in the document.")
                else:
                    st.warning("Please enter a question!")
        
        with col2:
            st.header("📋 Document Info")
            st.write(f"**Filename:** {st.session_state.filename}")
            
            # Show text preview
            st.subheader("📖 Text Preview")
            preview_text = st.session_state.document_text[:500] + "..." if len(st.session_state.document_text) > 500 else st.session_state.document_text
            st.text_area("", preview_text, height=300, disabled=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 How it works:
        
        1. **Upload** a PDF document using the sidebar
        2. **Wait** for text extraction to complete
        3. **Ask** questions about the document content
        4. **Get** AI-powered answers instantly!
        
        ### 📝 Example questions you can ask:
        - What is the main topic of this document?
        - What are the key requirements mentioned?
        - What is the deadline or timeline?
        - Who are the stakeholders involved?
        
        ### 🔧 Technical Details:
        - Uses DistilBERT model for question answering
        - Supports PDF text extraction
        - Provides confidence scores for answers
        """)

if __name__ == "__main__":
    main()
