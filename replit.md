# Excel Data Insights & KPI Analyzer

## Overview

This is a production-ready Streamlit-based web application that provides automated data analysis and KPI generation for Excel files. Users can upload Excel files and receive comprehensive insights including statistical analysis, data quality metrics, and focused business KPI recommendations. The application filters out invalid metrics (NaN/infinite values) and prioritizes critical KPIs for better user experience. Built with Python, it leverages data science libraries like pandas, numpy, scipy, and plotly for robust data processing and interactive visualization.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Layout**: Wide layout with expandable sidebar for file operations
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for maintaining user data across interactions

### Backend Architecture
- **Core Framework**: Python 3.11 with modular utility classes
- **Data Processing**: Pandas for data manipulation and analysis
- **Statistical Analysis**: Scipy for advanced statistical computations
- **File Handling**: Openpyxl for Excel file processing with multi-sheet support

### Module Structure
```
/utils/
├── data_analyzer.py    # Core data analysis functionality
├── kpi_generator.py    # Business KPI generation logic
└── visualizations.py  # Plotly-based visualization components
```

## Key Components

### Data Analyzer (`utils/data_analyzer.py`)
**Purpose**: Core data analysis engine
- Provides basic dataset information (rows, columns, memory usage)
- Generates detailed column analysis with data types and null counts
- Calculates summary statistics for numerical columns
- Implements data quality metrics including missing data and duplicate detection

**Rationale**: Centralized analysis logic ensures consistent data processing across the application and enables reusable analytical functions.

### KPI Generator (`utils/kpi_generator.py`)
**Purpose**: Intelligent KPI recommendation system
- Automatically suggests relevant KPIs based on data structure
- Generates statistical, performance, quality, time-based, and business KPIs
- Provides formula definitions and business value explanations
- Adapts recommendations to available data types (numerical, categorical, datetime)

**Rationale**: Automated KPI generation saves users time and provides expert-level insights without requiring deep analytical knowledge.

### Data Visualizer (`utils/visualizations.py`)
**Purpose**: Interactive visualization component
- Creates distribution plots with multiple chart types (histogram, box plot, density)
- Generates comprehensive visual analysis for numerical data
- Implements subplot layouts for multi-faceted data views

**Rationale**: Visual data exploration is crucial for understanding patterns and outliers that statistical summaries might miss.

## Data Flow

1. **File Upload**: User uploads Excel file through Streamlit file uploader
2. **Sheet Selection**: Multi-sheet Excel files allow user to select specific sheet
3. **Data Loading**: Pandas reads Excel data into DataFrame with error handling
4. **Analysis Pipeline**:
   - DataAnalyzer processes basic information and statistics
   - KPIGenerator creates tailored KPI recommendations
   - DataVisualizer prepares interactive charts
5. **Results Display**: Streamlit renders analysis results in organized tabs/sections
6. **State Persistence**: Analysis results stored in session state for navigation

## External Dependencies

### Core Libraries
- **streamlit (>=1.46.0)**: Web application framework
- **pandas (>=2.3.0)**: Data manipulation and analysis
- **numpy (>=2.3.1)**: Numerical computing foundation
- **plotly (>=6.1.2)**: Interactive visualization library
- **scipy (>=1.16.0)**: Statistical analysis functions
- **openpyxl (>=3.1.5)**: Excel file reading/writing

### Supporting Libraries
- **altair**: Additional visualization capabilities
- **cachetools**: Performance optimization for repeated calculations
- **jinja2, jsonschema**: Template and validation support

**Rationale**: Dependencies chosen for stability, performance, and comprehensive feature sets. Version pinning ensures reproducible deployments.

## Deployment Strategy

### Platform Configuration
- **Target**: Replit autoscale deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port**: 5000 (configured for Streamlit server)
- **Environment**: Stable Nix channel 24_05 for consistency

### Development Workflow
- **Run Command**: `streamlit run app.py --server.port 5000`
- **Development Mode**: Hot reload enabled for rapid iteration
- **Configuration**: Streamlit headless mode for server deployment

**Rationale**: Replit provides zero-configuration deployment with automatic scaling, eliminating infrastructure management overhead.

## Changelog

- June 26, 2025: Fixed Plotly visualization serialization errors and enhanced KPI filtering
- June 25, 2025: Initial setup

## User Preferences

- Preferred communication style: Simple, everyday language
- KPI Display: Only show critical KPIs (Business, Data Quality, Performance categories)
- Data Quality: Filter out NaN and infinite values from metrics display
- KPI Focus: Limit to top 3 KPIs per category for better user experience