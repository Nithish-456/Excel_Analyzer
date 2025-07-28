import pandas as pd
import numpy as np
import openai

class KPIGenerator:
    def __init__(self, data):
        self.data = data
        self.numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    def generate_kpi_suggestions(self):
        """Generate KPI suggestions based on data structure and content."""
        kpi_suggestions = {}
        
        # Basic Statistical KPIs
        if self.numerical_cols:
            kpi_suggestions['Statistical KPIs'] = self._generate_statistical_kpis()
        
        # Performance KPIs
        kpi_suggestions['Performance KPIs'] = self._generate_performance_kpis()
        
        # Quality KPIs
        kpi_suggestions['Data Quality KPIs'] = self._generate_quality_kpis()
        
        # Time-based KPIs
        if self.datetime_cols:
            kpi_suggestions['Time-based KPIs'] = self._generate_time_based_kpis()
        
        # Business-specific KPIs
        kpi_suggestions['Business KPIs'] = self._generate_business_kpis()
        
        return kpi_suggestions
    
    def _generate_statistical_kpis(self):
        """Generate statistical KPIs for numerical data."""
        kpis = []
        
        for col in self.numerical_cols:
            # Mean-based KPIs
            kpis.append({
                'name': f'Average {col}',
                'description': f'Average value of {col} across all records',
                'formula': f'SUM({col}) / COUNT({col})',
                'business_value': 'Provides central tendency measure for decision making',
                'calculation': lambda data, column=col: data[column].mean()
            })
            
            # Variance-based KPIs
            kpis.append({
                'name': f'{col} Variability Index',
                'description': f'Coefficient of variation for {col} to measure relative variability',
                'formula': f'STDEV({col}) / MEAN({col}) * 100',
                'business_value': 'Helps identify consistency and predictability in data',
                'calculation': lambda data, column=col: (data[column].std() / data[column].mean()) * 100 if data[column].mean() != 0 and not pd.isna(data[column].mean()) else None
            })
            
            # Growth/Change KPIs if multiple records exist
            if len(self.data) > 1:
                kpis.append({
                    'name': f'{col} Range Ratio',
                    'description': f'Ratio of maximum to minimum values in {col}',
                    'formula': f'MAX({col}) / MIN({col})',
                    'business_value': 'Indicates the spread and potential outliers in the data',
                    'calculation': lambda data, column=col: data[column].max() / data[column].min() if data[column].min() != 0 and not pd.isna(data[column].min()) else None
                })
        
        return kpis
    
    def _generate_performance_kpis(self):
        """Generate performance-related KPIs."""
        kpis = []
        
        # Data volume KPIs
        kpis.append({
            'name': 'Data Volume Index',
            'description': 'Total number of records in the dataset',
            'formula': 'COUNT(records)',
            'business_value': 'Measures data availability and sample size for analysis',
            'calculation': lambda data: len(data)
        })
        
        # Data density KPI
        kpis.append({
            'name': 'Data Density Score',
            'description': 'Percentage of non-null values across all fields',
            'formula': '(Total Non-Null Cells / Total Cells) * 100',
            'business_value': 'Indicates data completeness and reliability',
            'calculation': lambda data: ((data.count().sum()) / (data.shape[0] * data.shape[1])) * 100
        })
        
        # Unique value ratio for categorical columns
        for col in self.categorical_cols:
            kpis.append({
                'name': f'{col} Diversity Index',
                'description': f'Ratio of unique values to total values in {col}',
                'formula': f'UNIQUE_COUNT({col}) / COUNT({col})',
                'business_value': 'Measures diversity and granularity of categorical data',
                'calculation': lambda data, column=col: data[column].nunique() / len(data)
            })
        
        return kpis
    
    def _generate_quality_kpis(self):
        """Generate data quality KPIs."""
        kpis = []
        
        # Completeness KPI
        kpis.append({
            'name': 'Data Completeness Rate',
            'description': 'Percentage of complete records (no missing values)',
            'formula': 'Complete Records / Total Records * 100',
            'business_value': 'Ensures data reliability for business decisions',
            'calculation': lambda data: (len(data.dropna()) / len(data)) * 100
        })
        
        # Uniqueness KPI
        kpis.append({
            'name': 'Data Uniqueness Rate',
            'description': 'Percentage of unique records in the dataset',
            'formula': '(Total Records - Duplicate Records) / Total Records * 100',
            'business_value': 'Identifies data redundancy and quality issues',
            'calculation': lambda data: ((len(data) - data.duplicated().sum()) / len(data)) * 100
        })
        
        # Consistency KPI for numerical data
        if self.numerical_cols:
            kpis.append({
                'name': 'Numerical Data Consistency Score',
                'description': 'Average coefficient of variation across all numerical columns',
                'formula': 'AVERAGE(STDEV(column) / MEAN(column)) for all numerical columns',
                'business_value': 'Measures overall data consistency and reliability',
                'calculation': lambda data: np.mean([
                    (data[col].std() / data[col].mean()) if data[col].mean() != 0 and not pd.isna(data[col].mean()) else 0 
                    for col in data.select_dtypes(include=[np.number]).columns
                ]) if len(data.select_dtypes(include=[np.number]).columns) > 0 else None
            })
        
        return kpis
    
    def _generate_time_based_kpis(self):
        """Generate time-based KPIs if datetime columns exist."""
        kpis = []
        
        for col in self.datetime_cols:
            # Time span KPI
            kpis.append({
                'name': f'{col} Time Span',
                'description': f'Total time period covered by {col}',
                'formula': f'MAX({col}) - MIN({col})',
                'business_value': 'Indicates the temporal coverage of the dataset',
                'calculation': lambda data, column=col: (data[column].max() - data[column].min()).days if not pd.isna(data[column].max()) and not pd.isna(data[column].min()) else None
            })
            
            # Data frequency KPI
            kpis.append({
                'name': f'{col} Data Frequency',
                'description': f'Average number of records per day in {col}',
                'formula': f'COUNT(records) / DAYS_BETWEEN(MAX({col}), MIN({col}))',
                'business_value': 'Measures data collection frequency and consistency',
                'calculation': lambda data, column=col: len(data) / max(1, (data[column].max() - data[column].min()).days) if not pd.isna(data[column].max()) and not pd.isna(data[column].min()) else None
            })
        
        # If numerical data exists with time data
        if self.numerical_cols and self.datetime_cols:
            for num_col in self.numerical_cols:
                for date_col in self.datetime_cols:
                    kpis.append({
                        'name': f'{num_col} Growth Rate',
                        'description': f'Average change in {num_col} over time period in {date_col}',
                        'formula': f'(LAST_VALUE({num_col}) - FIRST_VALUE({num_col})) / FIRST_VALUE({num_col}) * 100',
                        'business_value': 'Tracks performance trends and growth patterns',
                        'calculation': lambda data, num_column=num_col, date_column=date_col: self._calculate_growth_rate(data, num_column, date_column)
                    })
        
        return kpis
    
    def _generate_business_kpis(self):
        """Generate business-specific KPIs based on common column names and patterns."""
        kpis = []
        
        # Revenue/Sales related KPIs
        revenue_cols = [col for col in self.data.columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'income', 'amount', 'value', 'price'])]
        for col in revenue_cols:
            if col in self.numerical_cols:
                kpis.append({
                    'name': f'Total {col}',
                    'description': f'Sum of all {col} values',
                    'formula': f'SUM({col})',
                    'business_value': 'Key performance indicator for business success',
                    'calculation': lambda data, column=col: data[column].sum()
                })
                
                kpis.append({
                    'name': f'Average {col} per Record',
                    'description': f'Average {col} value per transaction/record',
                    'formula': f'SUM({col}) / COUNT(records)',
                    'business_value': 'Measures average transaction value or efficiency',
                    'calculation': lambda data, column=col: data[column].mean()
                })
        
        # Count-based KPIs
        count_cols = [col for col in self.data.columns if any(keyword in col.lower() for keyword in ['count', 'quantity', 'qty', 'number', 'num'])]
        for col in count_cols:
            if col in self.numerical_cols:
                kpis.append({
                    'name': f'Total {col}',
                    'description': f'Sum of all {col} values',
                    'formula': f'SUM({col})',
                    'business_value': 'Tracks volume and operational metrics',
                    'calculation': lambda data, column=col: data[column].sum()
                })
        
        # Customer/User related KPIs
        customer_cols = [col for col in self.data.columns if any(keyword in col.lower() for keyword in ['customer', 'user', 'client', 'member'])]
        if customer_cols:
            kpis.append({
                'name': 'Customer/User Diversity',
                'description': 'Number of unique customers/users in the dataset',
                'formula': f'UNIQUE_COUNT({customer_cols[0]})',
                'business_value': 'Measures customer base size and market reach',
                'calculation': lambda data, column=customer_cols[0]: data[column].nunique()
            })
        
        # Conversion rate KPI if applicable columns exist
        if len([col for col in self.categorical_cols if any(keyword in col.lower() for keyword in ['status', 'result', 'outcome'])]) > 0:
            status_col = [col for col in self.categorical_cols if any(keyword in col.lower() for keyword in ['status', 'result', 'outcome'])][0]
            kpis.append({
                'name': f'{status_col} Success Rate',
                'description': f'Percentage of successful outcomes in {status_col}',
                'formula': f'COUNT(successful {status_col}) / COUNT(total {status_col}) * 100',
                'business_value': 'Measures success rate and operational efficiency',
                'calculation': lambda data, column=status_col: self._calculate_success_rate(data, column)
            })
        
        return kpis
    
    def _calculate_growth_rate(self, data, num_column, date_column):
        """Calculate growth rate for numerical data over time."""
        try:
            sorted_data = data.sort_values(date_column)
            first_value = sorted_data[num_column].iloc[0]
            last_value = sorted_data[num_column].iloc[-1]
            if first_value != 0 and not pd.isna(first_value) and not pd.isna(last_value):
                growth_rate = ((last_value - first_value) / first_value) * 100
                return growth_rate if not (np.isnan(growth_rate) or np.isinf(growth_rate)) else None
            return None
        except:
            return None
    
    def _calculate_success_rate(self, data, column):
        """Calculate success rate based on status column values."""
        try:
            total_count = len(data[column].dropna())
            if total_count == 0:
                return 0
            
            # Common success indicators
            success_keywords = ['success', 'complete', 'approved', 'yes', 'true', '1', 'active', 'confirmed']
            success_count = 0
            
            for value in data[column].dropna():
                if str(value).lower() in success_keywords:
                    success_count += 1
            
            return (success_count / total_count) * 100
        except:
            return 0
        
    def generate_custom_kpi_with_ai(self, user_prompt):
        """
        Generate a custom KPI using OpenAI based on user prompt and the current DataFrame.
        Returns a dictionary with KPI details (name, description, formula, business_value, calculation).
        """
        # Prepare context for OpenAI
        context = f"Data columns: {', '.join(self.data.columns)}\nSample data:\n{self.data.head().to_string()}"
        prompt = (
            f"You are a data analyst. Given the following data:\n{context}\n"
            f"Based on this, create a custom KPI as per the user's request: '{user_prompt}'. "
            "Return the KPI details as a JSON object with keys: name, description, formula, business_value, and a Python calculation code snippet using pandas DataFrame 'data'."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_content = response.choices[0].message.content
            import json
            import re
            match = re.search(r"\{.*\}", ai_content, re.DOTALL)
            if match:
                kpi_details = json.loads(match.group(0))
                return kpi_details
            else:
                return {"error": "Could not parse KPI details from AI response.", "raw_response": ai_content}
        except Exception as e:
            return {"error": str(e)}
