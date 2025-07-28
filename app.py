import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from utils.kpi_generator import KPIGenerator
from utils.visualizations import DataVisualizer
from utils.ppt_analyzer import PowerPointAnalyzer
import os
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Data Insights & Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'ppt_analyzer' not in st.session_state:
    st.session_state.ppt_analyzer = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

def initialize_openai_client():
    """Initialize OpenAI client with API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key != "dummy-key":
        return OpenAI(api_key=api_key)
    return None

def main():
    st.title("📊 Data Insights & Analysis Platform")
    st.markdown("Upload Excel files for data analysis or PowerPoint presentations for content insights.")
    
    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("📁 File Upload")
        
        # File type selection
        file_type = st.radio(
            "Select file type:",
            ["Excel Files", "PowerPoint Presentations"],
            help="Choose the type of file you want to analyze"
        )
        
        if file_type == "Excel Files":
            uploaded_file = st.file_uploader(
                "Choose an Excel file",
                type=['xlsx', 'xls'],
                help="Upload Excel files (.xlsx or .xls format)"
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a PowerPoint file",
                type=['pptx', 'ppt'],
                help="Upload PowerPoint files (.pptx or .ppt format)"
            )
        
        if uploaded_file is not None:
            try:
                if file_type == "Excel Files":
                    # Read Excel file and handle multiple sheets
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    
                    if len(sheet_names) > 1:
                        selected_sheet = st.selectbox(
                            "Select Sheet",
                            sheet_names,
                            help="Choose which sheet to analyze"
                        )
                    else:
                        selected_sheet = sheet_names[0]
                    
                    # Load data
                    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.session_state.data = data
                    st.session_state.ppt_analyzer = None
                    st.session_state.file_type = "excel"
                    st.session_state.analysis_complete = True
                    
                    st.success(f"✅ Successfully loaded sheet: {selected_sheet}")
                    st.info(f"📊 Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
                    
                else:
                    # Process PowerPoint file
                    file_content = io.BytesIO(uploaded_file.read())
                    ppt_analyzer = PowerPointAnalyzer(file_content)
                    
                    st.session_state.ppt_analyzer = ppt_analyzer
                    st.session_state.data = None
                    st.session_state.file_type = "powerpoint"
                    st.session_state.analysis_complete = True
                    
                    overview = ppt_analyzer.get_presentation_overview()
                    st.success(f"✅ Successfully loaded PowerPoint presentation")
                    st.info(f"📊 Presentation: {overview['total_slides']} slides, {overview['total_text_length']} characters")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.session_state.data = None
                st.session_state.ppt_analyzer = None
                st.session_state.analysis_complete = False
    
    # Main content area
    if st.session_state.data is not None and st.session_state.analysis_complete:
        data = st.session_state.data
        
        # Initialize analyzers
        kpi_generator = KPIGenerator(data)
        visualizer = DataVisualizer(data)
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 DataBot",
            "🎯 KPI Recommendations", 
            "📊 Visualizations", 
            "🔍 Correlation Analysis"
        ])
        
        with tab1:
            st.header("🤖 DataBot - Ask Questions About Your Data")
            
            # Initialize OpenAI client if not already done
            if st.session_state.openai_client is None:
                st.session_state.openai_client = initialize_openai_client()
            
            if st.session_state.openai_client is None:
                st.warning("⚠️ OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable to use DataBot.")
                st.info("DataBot can answer questions about your data, provide insights, and help with analysis.")
            else:
                # Chat interface
                st.subheader("💬 Chat with your data")
                
                # Display chat history
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.container():
                        st.write(f"**You:** {question}")
                        st.write(f"**DataBot:** {answer}")
                        st.divider()
                
                # Input for new question
                user_question = st.text_input(
                    "Ask a question about your data:",
                    placeholder="e.g., What are the main trends in this data? What insights can you provide?",
                    key="user_question"
                )
                
                if st.button("Ask DataBot") and user_question:
                    with st.spinner("DataBot is analyzing your data..."):
                        try:
                            # Prepare data context for AI
                            data_context = prepare_data_context(data)
                            
                            # Generate AI response
                            response = get_ai_response(st.session_state.openai_client, user_question, data_context)
                            
                            # Add to chat history
                            st.session_state.chat_history.append((user_question, response))
                            
                            # Clear input and rerun to show new message
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error getting response from DataBot: {str(e)}")
                
                # Clear chat history button
                if st.session_state.chat_history and st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with tab2:
            st.header("KPI Recommendations")
            
            kpi_suggestions = kpi_generator.generate_kpi_suggestions()
            
            if kpi_suggestions:
                # Filter and show only critical KPIs
                critical_categories = ['Business KPIs', 'Data Quality KPIs', 'Performance KPIs']
                
                for category in critical_categories:
                    if category in kpi_suggestions:
                        kpis = kpi_suggestions[category]
                        if kpis:  # Only show if there are KPIs in this category
                            st.subheader(f"{category}")
                            
                            # Limit to top 3 KPIs per category for focus
                            for kpi in kpis[:3]:
                                with st.expander(f"🎯 {kpi['name']}"):
                                    st.write(f"**Description:** {kpi['description']}")
                                    st.write(f"**Formula:** {kpi['formula']}")
                                    st.write(f"**Business Value:** {kpi['business_value']}")
                                    
                                    # Calculate KPI if possible
                                    if kpi.get('calculation'):
                                        try:
                                            result = kpi['calculation'](data)
                                            # Only show valid results (not NaN or infinite)
                                            if isinstance(result, (int, float)) and not (np.isnan(result) or np.isinf(result)):
                                                st.metric("Calculated Value", f"{result:.2f}")
                                            elif not isinstance(result, (int, float)):
                                                st.metric("Calculated Value", str(result))
                                        except Exception as e:
                                            st.warning(f"Could not calculate: {str(e)}")
            else:
                st.info("Upload data to see KPI recommendations.")
        
        with tab3:
            st.header("Data Visualizations")
            
            # Numerical columns distribution
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Filter columns that have valid data
            valid_numerical_cols = []
            for col in numerical_cols:
                clean_series = pd.to_numeric(data[col], errors='coerce')
                clean_data = clean_series.dropna()
                if len(clean_data) > 0:
                    valid_numerical_cols.append(col)
            
            if valid_numerical_cols:
                st.subheader("Numerical Data Distributions")
                selected_num_col = st.selectbox("Select numerical column", valid_numerical_cols)
                if selected_num_col:
                    fig = visualizer.create_distribution_plot(selected_num_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid numerical data available for distribution plots.")
            
            # Categorical columns
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            # Filter columns that have valid data and reasonable unique values
            valid_categorical_cols = []
            for col in categorical_cols:
                clean_data = data[col].dropna()
                if len(clean_data) > 0 and clean_data.nunique() > 1 and clean_data.nunique() <= 50:
                    valid_categorical_cols.append(col)
            
            if valid_categorical_cols:
                st.subheader("Categorical Data Analysis")
                selected_cat_col = st.selectbox("Select categorical column", valid_categorical_cols)
                if selected_cat_col:
                    fig = visualizer.create_categorical_plot(selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No suitable categorical data available for visualization.")
            
            # Time series analysis if datetime columns exist
            datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            # Filter datetime columns that have valid data
            valid_datetime_cols = []
            for col in datetime_cols:
                clean_series = pd.to_datetime(data[col], errors='coerce')
                clean_data = clean_series.dropna()
                if len(clean_data) > 1 and clean_data.nunique() > 1:
                    valid_datetime_cols.append(col)
            
            if valid_datetime_cols and valid_numerical_cols:
                st.subheader("Time Series Analysis")
                selected_date_col = st.selectbox("Select datetime column", valid_datetime_cols)
                if selected_date_col:
                    selected_value_col = st.selectbox("Select value column", valid_numerical_cols)
                    if selected_value_col:
                        fig = visualizer.create_time_series_plot(selected_date_col, selected_value_col)
                        st.plotly_chart(fig, use_container_width=True)
            elif valid_datetime_cols and not valid_numerical_cols:
                st.info("Time series analysis requires valid numerical data.")
            elif not valid_datetime_cols and valid_numerical_cols:
                st.info("Time series analysis requires valid datetime data.")
            else:
                st.info("Time series analysis requires both valid datetime and numerical data.")
        
        with tab4:
            st.header("Correlation Analysis")
            
            if len(valid_numerical_cols) > 1:
                # Filter data to only include valid numerical columns
                valid_data = data[valid_numerical_cols].copy()
                for col in valid_numerical_cols:
                    valid_data[col] = pd.to_numeric(valid_data[col], errors='coerce')
                valid_data = valid_data.dropna()
                
                if len(valid_data) > 1 and valid_data.shape[1] > 1:
                    correlation_matrix = valid_data.corr()
                    
                    # Correlation heatmap
                    fig = visualizer.create_correlation_heatmap(correlation_matrix)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strong correlations
                    st.subheader("Strong Correlations")
                    # Create temporary analyzer for filtered data
                    temp_analyzer = DataAnalyzer(valid_data)
                    strong_corr = temp_analyzer.find_strong_correlations()
                    if not strong_corr.empty:
                        st.dataframe(strong_corr, use_container_width=True)
                    else:
                        st.info("No strong correlations found (threshold: 0.7)")
                else:
                    st.info("Insufficient valid data for correlation analysis.")
            else:
                st.info("Need at least 2 valid numerical columns for correlation analysis.")
        
        with tab6:
            st.header("Export Analysis Report")
            
            if st.button("📄 Generate Report"):
                report = generate_analysis_report(data, analyzer, kpi_generator)
                
                # Create download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"data_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Display report preview
                st.subheader("Report Preview")
                st.text_area("Report Content", report, height=400)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to Excel Data Insights & KPI Analyzer! 🎉
        
        This application helps you analyze your Excel data and provides:
        
        - 📊 **Comprehensive Data Overview**: Basic statistics, data types, and structure analysis
        - 🔍 **Deep Data Insights**: Missing values, outliers, and data quality assessment
        - 🎯 **Smart KPI Recommendations**: Automated suggestions based on your data structure
        - 📈 **Interactive Visualizations**: Charts and graphs to understand your data better
        - 🔗 **Correlation Analysis**: Discover relationships between variables
        - 📄 **Exportable Reports**: Download comprehensive analysis reports
        
        ### How to Get Started:
        1. Click "Browse files" in the sidebar
        2. Upload your Excel file (.xlsx or .xls)
        3. Select a sheet if your file has multiple sheets
        4. Explore the analysis tabs above!
        
        ### Supported Features:
        - ✅ Multiple Excel formats (.xlsx, .xls)
        - ✅ Multi-sheet workbooks
        - ✅ Large dataset handling
        - ✅ Real-time analysis
        - ✅ Interactive visualizations
        """)

def generate_analysis_report(data, analyzer, kpi_generator):
    """Generate a comprehensive text report of the analysis."""
    report = []
    report.append("=" * 60)
    report.append("EXCEL DATA ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Basic Info
    report.append("1. BASIC INFORMATION")
    report.append("-" * 25)
    basic_info = analyzer.get_basic_info()
    for key, value in basic_info.items():
        report.append(f"{key}: {value}")
    report.append("")
    
    # Data Quality
    report.append("2. DATA QUALITY METRICS")
    report.append("-" * 30)
    quality_metrics = analyzer.get_data_quality_metrics()
    for metric, value in quality_metrics.items():
        report.append(f"{metric}: {value}")
    report.append("")
    
    # Missing Values
    report.append("3. MISSING VALUES ANALYSIS")
    report.append("-" * 35)
    missing_data = analyzer.analyze_missing_values()
    if not missing_data.empty:
        for idx, row in missing_data.iterrows():
            report.append(f"{idx}: {row['Missing_Count']} missing ({row['Missing_Percentage']:.2f}%)")
    else:
        report.append("No missing values found.")
    report.append("")
    
    # KPI Recommendations
    report.append("4. KPI RECOMMENDATIONS")
    report.append("-" * 25)
    kpi_suggestions = kpi_generator.generate_kpi_suggestions()
    for category, kpis in kpi_suggestions.items():
        report.append(f"\n{category.upper()}:")
        for kpi in kpis:
            report.append(f"  • {kpi['name']}: {kpi['description']}")
    
    return "\n".join(report)

def prepare_data_context(data):
    """Prepare data context for AI analysis."""
    context = {
        'shape': f"{data.shape[0]} rows × {data.shape[1]} columns",
        'columns': list(data.columns),
        'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
        'sample_data': data.head(5).to_dict('records'),
        'summary_stats': {}
    }
    
    # Add summary statistics for numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        try:
            stats = data[col].describe()
            context['summary_stats'][col] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'missing_count': int(data[col].isnull().sum())
            }
        except:
            continue
    
    return context

def get_ai_response(client, question, data_context):
    """Get AI response for user question about the data."""
    system_prompt = """You are DataBot, an expert data analyst assistant. You help users understand and analyze their Excel data by answering questions and providing insights.

You have access to information about the user's dataset including:
- Dataset structure (rows, columns, data types)
- Sample data
- Summary statistics for numerical columns
- Column names and types

Provide helpful, accurate, and actionable insights based on the data context provided. Keep responses concise but informative. If you need more specific data to answer a question, let the user know what additional information would be helpful."""

    user_prompt = f"""
    Dataset Context:
    - Shape: {data_context['shape']}
    - Columns: {', '.join(data_context['columns'])}
    - Data Types: {data_context['dtypes']}
    - Sample Data: {data_context['sample_data']}
    - Summary Statistics: {data_context['summary_stats']}
    
    User Question: {question}
    
    Please analyze this data and provide a helpful response to the user's question.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

if __name__ == "__main__":
    main()



