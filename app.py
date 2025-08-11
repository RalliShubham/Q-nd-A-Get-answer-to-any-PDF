import streamlit as st
import os
from pypdf import PdfReader
import tempfile
import re

# Set page config
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="wide"
)

# Cache the model loading with a better model
@st.cache_resource
def load_qa_model():
    try:
        from transformers import pipeline
        # Using a better model that gives longer, more complete answers
        return pipeline(
            task="question-answering", 
            model="deepset/roberta-base-squad2",  # Better model for longer answers
            tokenizer="deepset/roberta-base-squad2"
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Add space between numbers and letters
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Add space between letters and numbers
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extract and clean text from uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Read the PDF
        reader = PdfReader(tmp_file_path)
        document_text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Clean the text from this page
                    cleaned_page_text = clean_text(page_text)
                    document_text += cleaned_page_text + " "
            except Exception as e:
                st.warning(f"Could not extract text from page {page_num + 1}")
                continue
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Final cleaning
        final_text = clean_text(document_text)
        return final_text
        
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def find_relevant_context(question, full_text, max_length=2000):
    """Find the most relevant part of the document for the question"""
    if not full_text or len(full_text) <= max_length:
        return full_text
    
    # Convert question to lowercase for matching
    question_lower = question.lower()
    question_words = question_lower.split()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', full_text)
    
    # Score sentences based on question word matches
    scored_sentences = []
    for sentence in sentences:
        if len(sentence.strip()) < 10:  # Skip very short sentences
            continue
            
        sentence_lower = sentence.lower()
        score = 0
        
        # Count question word matches
        for word in question_words:
            if len(word) > 2:  # Skip very short words
                score += sentence_lower.count(word)
        
        scored_sentences.append((score, sentence.strip()))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Build context from top-scoring sentences
    context = ""
    for score, sentence in scored_sentences:
        if len(context) + len(sentence) > max_length:
            break
        if score > 0:  # Only include sentences with some relevance
            context += sentence + ". "
    
    # If no relevant sentences found, use the beginning of the document
    if not context.strip():
        context = full_text[:max_length]
    
    return context.strip()

def get_answer(question, context, qa_pipeline):
    """Get answer from the QA pipeline with better processing"""
    if qa_pipeline is None:
        return None
        
    try:
        # Find relevant context
        relevant_context = find_relevant_context(question, context, max_length=1500)
        
        if not relevant_context:
            return None
        
        # Get answer from model
        result = qa_pipeline(
            question=question, 
            context=relevant_context,
            max_answer_len=100,  # Allow longer answers
            handle_impossible_answer=True
        )
        
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
    if 'qa_model' not in st.session_state:
        st.session_state.qa_model = None
    
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
                with st.spinner("Extracting and processing text from PDF..."):
                    st.session_state.document_text = extract_text_from_pdf(uploaded_file)
                
                if st.session_state.document_text:
                    st.success("✅ PDF processed successfully!")
                    st.info(f"📊 Extracted {len(st.session_state.document_text)} characters")
                    
                    # Show a sample of cleaned text
                    with st.expander("📖 View extracted text sample"):
                        sample_text = st.session_state.document_text[:500] + "..." if len(st.session_state.document_text) > 500 else st.session_state.document_text
                        st.text(sample_text)
    
    # Main content area
    if st.session_state.document_text:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🤖 Ask Questions")
            
            # Load QA model only when needed
            if st.session_state.qa_model is None:
                with st.spinner("Loading AI model... This may take a moment on first load."):
                    st.session_state.qa_model = load_qa_model()
            
            if st.session_state.qa_model is None:
                st.error("❌ Could not load the AI model. Please try refreshing the page.")
                return
            
            # Provide example questions
            st.markdown("**💡 Try these example questions:**")
            example_questions = [
                "What is the policy on PTO?",
                "What are the working hours?",
                "What is the notice period for resignation?",
                "What are the benefits provided?",
                "What is the dress code policy?"
            ]
            
            selected_example = st.selectbox("Choose an example question:", [""] + example_questions)
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                value=selected_example if selected_example else "",
                placeholder="e.g., What is the policy on paid time off?",
                key="question_input"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question.strip():
                    with st.spinner("Analyzing document and finding answer..."):
                        result = get_answer(question, st.session_state.document_text, st.session_state.qa_model)
                    
                    if result and result.get('answer'):
                        st.subheader("💡 Answer:")
                        
                        # Display the answer with better formatting
                        answer = result['answer'].strip()
                        st.markdown(f"**{answer}**")
                        
                        confidence = round(result.get('score', 0) * 100, 1)
                        
                        # Color-code confidence
                        if confidence >= 70:
                            st.success(f"✅ High Confidence: {confidence}%")
                        elif confidence >= 40:
                            st.warning(f"⚠️ Medium Confidence: {confidence}%")
                        else:
                            st.error(f"❌ Low Confidence: {confidence}%")
                            st.info("💡 Try rephrasing your question or check if the information exists in the document.")
                        
                        # Show the context that was used
                        with st.expander("📄 View relevant text section"):
                            relevant_context = find_relevant_context(question, st.session_state.document_text)
                            st.text_area("Context used for answer:", relevant_context, height=150)
                            
                    else:
                        st.error("❌ Could not find an answer to your question in the document.")
                        st.info("💡 Try rephrasing your question or make sure the information exists in the document.")
                else:
                    st.warning("Please enter a question!")
        
        with col2:
            st.header("📋 Document Info")
            st.write(f"**Filename:** {st.session_state.filename}")
            st.write(f"**Text Length:** {len(st.session_state.document_text):,} characters")
            
            # Show text preview
            st.subheader("📖 Text Preview")
            preview_text = st.session_state.document_text[:300] + "..." if len(st.session_state.document_text) > 300 else st.session_state.document_text
            st.text_area("", preview_text, height=200, disabled=True)
            
            # Tips for better questions
            st.subheader("💡 Tips for Better Results")
            st.markdown("""
            - Ask specific questions
            - Use keywords from the document
            - Try different phrasings
            - Ask about one topic at a time
            """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 How it works:
        
        1. **Upload** a PDF document using the sidebar
        2. **Wait** for text extraction and processing
        3. **Ask** specific questions about the document content
        4. **Get** AI-powered answers with confidence scores!
        
        ### 📝 Example questions you can ask:
        - What is the policy on paid time off?
        - What are the working hours mentioned?
        - What is the notice period for resignation?
        - What benefits are provided to employees?
        - What is the dress code policy?
        
        ### 🔧 Features:
        - ✅ Advanced text extraction and cleaning
        - ✅ Smart context selection for better answers
        - ✅ Confidence scoring
        - ✅ Relevant text highlighting
        
        ### 🚀 Get Started:
        Upload a PDF file using the sidebar to begin!
        """)

if __name__ == "__main__":
    main()
