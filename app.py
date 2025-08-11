import streamlit as st

# Set page config
st.set_page_config(
    page_title="Text Question Answering",
    page_icon="📝",
    layout="wide"
)

# Try to import transformers with error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing transformers: {e}")
    st.error("Please make sure transformers is properly installed")
    TRANSFORMERS_AVAILABLE = False

# Show loading animation immediately
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

if not TRANSFORMERS_AVAILABLE:
    st.error("Transformers library is not available. Please check your requirements.txt file.")
    st.stop()

if not st.session_state.app_loaded:
    # Show loading screen
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 50vh;">
        <div style="text-align: center;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
            <h3 style="margin-top: 20px;">Loading Application...</h3>
            <p>Please wait while we initialize the system</p>
        </div>
    </div>
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load the model in the background
    with st.spinner("Initializing AI model..."):
        try:
            qa_model = pipeline(task='question-answering', model='deepset/roberta-base-squad2')
            st.session_state.qa_model = qa_model
            st.session_state.app_loaded = True
            st.rerun()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("This might be due to insufficient memory or network issues on Streamlit Cloud")
            st.stop()

# Cache the model loading
@st.cache_resource
def load_qa_model():
    try:
        return pipeline(task='question-answering', model='deepset/roberta-base-squad2')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Document Question Answering System")
    st.markdown("Analyze your documents and get answers to your questions using advanced AI")
    
    # Initialize session state
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Sidebar for text input
    with st.sidebar:
        st.header("Document Input")
        
        # Text input area
        user_text = st.text_area(
            "Paste your document text here:",
            value=st.session_state.document_text,
            height=400,
            placeholder="Paste your document text here and ask questions about its content.",
            help="Copy and paste the text from your document here"
        )
        
        # Update button
        if st.button("Load Text", type="primary"):
            if user_text.strip():
                st.session_state.document_text = user_text.strip()
                st.success("Text loaded successfully")
                st.info(f"Characters: {len(st.session_state.document_text)}")
                st.info(f"Words: {len(st.session_state.document_text.split())}")
            else:
                st.warning("Please paste some text first")
        
        # Text statistics
        if st.session_state.document_text:
            st.markdown("---")
            st.markdown("**Document Statistics:**")
            st.info(f"Characters: {len(st.session_state.document_text):,}")
            st.info(f"Words: {len(st.session_state.document_text.split()):,}")
    
    # Main content area
    if st.session_state.document_text:
        # Create tabs
        tab1, tab2 = st.tabs(["Question & Answer", "Document View"])
        
        with tab1:
            st.success("AI model loaded and ready")
            
            st.header("Ask Your Question")
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                value=st.session_state.current_question,
                placeholder="Type your question about the document...",
                key="question_input"
            )
            
            # Update session state when question changes
            if question != st.session_state.current_question:
                st.session_state.current_question = question
            
            if st.button("Get Answer", type="primary", use_container_width=True):
                if st.session_state.current_question.strip():
                    st.info(f"Analyzing: {st.session_state.current_question}")
                    
                    with st.spinner("Processing your question..."):
                        try:
                            prediction = st.session_state.qa_model(
                                question=st.session_state.current_question, 
                                context=st.session_state.document_text
                            )
                            
                            st.markdown("---")
                            st.subheader("Answer:")
                            
                            # Answer display
                            st.markdown(f"""
                            <div style="
                                background-color: #f0f2f6;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 5px solid #1f77b4;
                                margin: 10px 0;
                            ">
                                <h3 style="margin-top: 0; color: #1f77b4;">
                                    {prediction['answer']}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence metrics
                            confidence = round(prediction['score'] * 100, 1)
                            
                            col_conf1, col_conf2, col_conf3 = st.columns(3)
                            with col_conf1:
                                st.metric("Confidence", f"{confidence}%")
                            
                            with col_conf2:
                                if confidence >= 60:
                                    st.success("High Confidence")
                                elif confidence >= 30:
                                    st.warning("Medium Confidence")
                                else:
                                    st.error("Low Confidence")
                            
                            with col_conf3:
                                st.metric("Answer Length", f"{len(prediction['answer'])} chars")
                            
                            # Context view
                            answer_text = prediction['answer']
                            if answer_text in st.session_state.document_text:
                                start_pos = st.session_state.document_text.find(answer_text)
                                context_start = max(0, start_pos - 200)
                                context_end = min(len(st.session_state.document_text), start_pos + len(answer_text) + 200)
                                context_snippet = st.session_state.document_text[context_start:context_end]
                                
                                with st.expander("View answer in context"):
                                    highlighted_context = context_snippet.replace(answer_text, f"**{answer_text}**")
                                    st.markdown(highlighted_context)
                            
                            # Technical details
                            with st.expander("Technical Details"):
                                st.json(prediction)
                            
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                else:
                    st.warning("Please enter a question")
        
        with tab2:
            st.header("Document Content")
            st.info(f"Document length: {len(st.session_state.document_text)} characters")
            
            # Search functionality
            search_term = st.text_input("Search in document:", placeholder="Enter text to highlight...")
            
            if search_term and search_term.strip():
                # Create highlighted version using HTML
                highlighted_text = st.session_state.document_text.replace(
                    search_term, 
                    f'<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">{search_term}</mark>'
                )
                
                st.markdown("**Document with highlighted search results:**")
                st.markdown(
                    f'<div style="background-color: white; padding: 20px; border-radius: 5px; border: 1px solid #ddd; height: 500px; overflow-y: scroll; font-family: monospace; white-space: pre-wrap;">{highlighted_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.text_area(
                    "Full document:",
                    st.session_state.document_text,
                    height=500,
                    disabled=True
                )
            
            # Download option
            st.download_button(
                label="Download as text file",
                data=st.session_state.document_text,
                file_name="document.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        # Instructions
        st.markdown("""
        ## Getting Started
        
        ### Features:
        - Advanced natural language processing using RoBERTa model
        - Confidence scoring for answer reliability
        - Context highlighting to show source of answers
        - Document search and download capabilities
        
        ### How to use:
        1. Paste your document text in the sidebar
        2. Click "Load Text" to process the document
        3. Type your question in the main area
        4. Get answers with confidence scores
        
        ### Tips:
        - Longer documents provide better context for answers
        - Ask specific questions about information in your document
        - Check confidence scores to evaluate answer reliability
        - Use the context view to verify answers
        """)
        
        # Example
        with st.expander("Example Document Text"):
            st.code("""
Employee Handbook - Working Conditions and Benefits

Working Hours:
The company is open Monday through Friday from 9:00 AM to 6:00 PM.
Saturday hours are 10:00 AM to 4:00 PM. The office is closed on Sundays.

Benefits:
Full-time employees are eligible for health insurance after 90 days.
Paid time off accrues at 1.5 days per month for the first year.
The company observes 10 federal holidays annually.

Compensation:
Salary reviews are conducted annually in January.
Overtime is paid at 1.5 times the regular rate for hours over 40 per week.
            """)

if __name__ == "__main__":
    main()
