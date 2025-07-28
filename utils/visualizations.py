import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, data):
        self.data = data
    
    def create_distribution_plot(self, column):
        """Create distribution plot for numerical column."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{column} - Histogram',
                f'{column} - Box Plot',
                f'{column} - Density Plot',
                f'{column} - Summary Statistics'
            ),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Clean data and convert to numeric
        clean_series = pd.to_numeric(self.data[column], errors='coerce')
        clean_data = clean_series.dropna()
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=clean_data, name="Distribution", showlegend=False),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=clean_data, name="Box Plot", showlegend=False),
            row=1, col=2
        )
        
        # Density plot (using histogram with density)
        fig.add_trace(
            go.Histogram(
                x=clean_data, 
                histnorm='probability density',
                name="Density",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Summary statistics table
        stats = clean_data.describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[
                    [str(idx) for idx in stats.index],
                    [f"{float(val):.2f}" for val in stats.values]
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_categorical_plot(self, column):
        """Create visualization for categorical column."""
        # Clean data and get value counts
        clean_data = self.data[column].dropna()
        
        if len(clean_data) == 0:
            fig = go.Figure()
            fig.update_layout(title=f"No valid data in {column}")
            return fig
        
        value_counts = clean_data.value_counts()
        
        # Limit to top 20 categories to avoid cluttered visualization
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{column} - Bar Chart (Top {len(value_counts)})', f'{column} - Pie Chart'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Convert to strings to avoid serialization issues
        labels = [str(label) for label in value_counts.index]
        values = [int(val) for val in value_counts.values]
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name="Count",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Distribution",
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"Categorical Analysis: {column}",
            height=400
        )
        
        return fig
    
    def create_time_series_plot(self, date_column, value_column):
        """Create time series plot."""
        # Sort data by date and clean
        sorted_data = self.data.sort_values(date_column).dropna(subset=[date_column, value_column])
        
        fig = go.Figure()
        
        # Convert to proper types and align data
        x_data = pd.to_datetime(sorted_data[date_column])
        y_data = pd.to_numeric(sorted_data[value_column], errors='coerce')
        
        # Create aligned dataset without NaN values
        plot_data = pd.DataFrame({
            'date': x_data,
            'value': y_data
        }).dropna()
        
        if len(plot_data) == 0:
            fig.update_layout(title=f"No valid data for {value_column} over {date_column}")
            return fig
        
        # Line plot
        fig.add_trace(
            go.Scatter(
                x=plot_data['date'],
                y=plot_data['value'],
                mode='lines+markers',
                name=f'{value_column} over time',
                line=dict(width=2),
                marker=dict(size=4)
            )
        )
        
        # Add trend line if enough data points
        if len(plot_data) > 2:
            try:
                z = np.polyfit(
                    pd.to_numeric(plot_data['date']), 
                    plot_data['value'], 
                    1
                )
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['date'],
                        y=p(pd.to_numeric(plot_data['date'])),
                        mode='lines',
                        name='Trend',
                        line=dict(dash='dash', color='red')
                    )
                )
            except:
                pass  # Skip trend line if calculation fails
        
        fig.update_layout(
            title=f'{value_column} over {date_column}',
            xaxis_title=date_column,
            yaxis_title=value_column,
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap."""
        # Convert to proper format for serialization
        z_values = correlation_matrix.values.astype(float)
        x_labels = [str(col) for col in correlation_matrix.columns]
        y_labels = [str(idx) for idx in correlation_matrix.index]
        text_values = correlation_matrix.round(3).values.astype(float)
        
        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale='RdBu',
                zmid=0,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title="Correlation Matrix Heatmap",
            height=600,
            width=800
        )
        
        return fig
    
    def create_scatter_plot(self, x_column, y_column, color_column=None):
        """Create scatter plot between two numerical columns."""
        # Clean the data first
        plot_data = self.data[[x_column, y_column]].copy()
        plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors='coerce')
        plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors='coerce')
        plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            fig = go.Figure()
            fig.update_layout(title=f"No valid data for {y_column} vs {x_column}")
            return fig
        
        if color_column and color_column in self.data.columns:
            plot_data[color_column] = self.data[color_column]
            fig = px.scatter(
                plot_data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f'{y_column} vs {x_column} (colored by {color_column})'
            )
        else:
            fig = px.scatter(
                plot_data,
                x=x_column,
                y=y_column,
                title=f'{y_column} vs {x_column}'
            )
        
        # Add trend line safely
        try:
            trend_fig = px.scatter(
                plot_data, 
                x=x_column, 
                y=y_column, 
                trendline="ols"
            )
            if len(trend_fig.data) > 1:
                fig.add_traces(trend_fig.data[1])
        except:
            pass  # Skip trend line if calculation fails
        
        return fig
    
    def create_multi_column_comparison(self, columns):
        """Create comparison plot for multiple numerical columns."""
        fig = go.Figure()
        
        for col in columns:
            if col in self.data.select_dtypes(include=[np.number]).columns:
                # Clean and normalize data for comparison
                clean_col_series = pd.to_numeric(self.data[col], errors='coerce')
                clean_col_data = clean_col_series.dropna()
                if len(clean_col_data) > 0 and clean_col_data.max() != clean_col_data.min():
                    normalized_data = (clean_col_data - clean_col_data.min()) / (clean_col_data.max() - clean_col_data.min())
                    
                    fig.add_trace(
                        go.Scatter(
                            y=normalized_data.values,
                            mode='lines',
                            name=str(col)
                        )
                    )
        
        fig.update_layout(
            title="Normalized Comparison of Multiple Columns",
            xaxis_title="Index",
            yaxis_title="Normalized Value (0-1)",
            height=500
        )
        
        return fig
    
    def create_outlier_visualization(self, column):
        """Create visualization highlighting outliers."""
        # Clean and convert data
        clean_series = pd.to_numeric(self.data[column], errors='coerce')
        clean_data = clean_series.dropna()
        
        if len(clean_data) == 0:
            # Return empty figure if no valid data
            fig = go.Figure()
            fig.update_layout(title=f"No valid numerical data in {column}")
            return fig
        
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create colors for outliers
        colors = ['red' if x < lower_bound or x > upper_bound else 'blue' for x in clean_data]
        
        fig = go.Figure()
        
        # Scatter plot with outliers highlighted
        fig.add_trace(
            go.Scatter(
                x=list(range(len(clean_data))),
                y=clean_data.values,
                mode='markers',
                marker=dict(color=colors),
                name='Data Points',
                text=[f'Index: {i}<br>Value: {float(val):.2f}<br>{"Outlier" if c == "red" else "Normal"}' 
                      for i, (val, c) in enumerate(zip(clean_data.values, colors))],
                hovertemplate='%{text}<extra></extra>'
            )
        )
        
        # Add boundary lines
        fig.add_hline(y=float(lower_bound), line_dash="dash", line_color="red", 
                     annotation_text="Lower Bound")
        fig.add_hline(y=float(upper_bound), line_dash="dash", line_color="red", 
                     annotation_text="Upper Bound")
        
        fig.update_layout(
            title=f"Outlier Detection: {column}",
            xaxis_title="Index",
            yaxis_title=column,
            height=500
        )
        
        return fig
