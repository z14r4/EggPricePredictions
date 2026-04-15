# Import packages
from dash import Dash, html, dash_table, dcc
import pandas as pd
import numpy as np
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Archive.backtest import feature_importance

# Incorporate data
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
df = pd.read_csv('df.csv')

egg = pd.read_csv("y.csv")

# Join X and y on their indices
X['egg_price'] = egg

# Add a new column 'Y' which is the 'egg_price' column shifted down by 1 entry
X['Y'] = X['egg_price'].shift(-1)

X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Y', 'Year-Month']), X['Y'], test_size=0.25, shuffle = False)

# Precomputed summaries
price_summary = df['Price'].describe().reset_index()

# Ensure 'Year-Month' is in datetime format for correlation calculation
X['Year-Month'] = pd.to_datetime(X['Year-Month'], format='%Y-%m', errors='coerce')

# Drop non-numeric columns before calculating the correlation matrix
corr_matrix = X_train.corr()

# Calculate the percentage increase from the previous month's price
latest_price = y.iloc[-1]
previous_price = y.iloc[-2]
percent_increase = (((latest_price - previous_price) / previous_price) * 100).iloc[0]

feature_importance = feature_importance.drop("const")

#R2 and RMSE values pre-backtest
# Models
models = ["Linear Regression", "Random Forest"]

# R² values
r2_before = [0.526, -0.053]
r2_after = [0.658, 0.171]

# RMSE values
rmse_before = [0.496, 0.740]
rmse_after = [0.177, 0.431]

# Build the figure for Pre Backtest Performance
fig_pre = make_subplots(
    rows=1, cols=2,
    subplot_titles=("R² Scores", "RMSE Scores"),
    shared_xaxes=False,
    horizontal_spacing=0.15
)

# R²
fig_pre.add_trace(go.Bar(name="Before CV", x=models, y=r2_before, marker_color='lightblue'), row=1, col=1).add_trace(go.Bar(name="After CV", x=models, y=r2_after, marker_color='steelblue'), row=1, col=1)

# RMSE
fig_pre.add_trace(go.Bar(name="Before CV", x=models, y=rmse_before, marker_color='lightcoral'), row=1, col=2).add_trace(go.Bar(name="After CV", x=models, y=rmse_after, marker_color='indianred'), row=1, col=2)

fig_pre.update_layout(
    barmode='group',
    height=450,
    title_text="Before vs After CV R² Scores & RMSE Scores (Pre Backtesting)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    )
)
fig_pre.update_yaxes(title_text="R² Score", row=1, col=1).update_yaxes(title_text="RMSE", row=1, col=2).update_xaxes(title_text="Model", row=1, col=1).update_xaxes(title_text="Model", row=1, col=2)

# Initialize the app
app = Dash()
server = app.server

# App layout
app.layout = html.Div([
    html.Div('Predicting Egg Prices', style={'textAlign': 'left', 'color': 'black', 'font': 'Poppins', 'fontSize': 100, 'marginBottom': '20px'}),
    
    # First row: Price Cards
    html.Div([
        html.Div([
            html.H5('Latest Price', style={'textAlign': 'left', 'fontSize': 25, 'fontWeight': 'bold'}),
            html.Div(f"${df['Price'].iloc[-1]:,.2f}", style={'textAlign': 'center', 'fontSize': 75, 'color': 'green', 'minHeight': '100px'}),
            html.Small(f"↑ {percent_increase:.2f}% from last month", style={'display': 'block', 'textAlign': 'center', 'color': 'green', 'minHeight': '50px'})
        ], style={'width': '50%', 'padding': '10px', 'border': '1px solid lightgray', 'borderRadius': '10px', 'margin': '10px'}),
        html.Div([
            html.H5('Price Range', style={'textAlign': 'left','fontSize': 25, 'fontWeight': 'bold'}),
            html.Div(f"${df['Price'].min():.0f} - ${df['Price'].max():.0f}", style={'textAlign': 'center', 'fontSize': 40, 'color': 'orange'}),
        ], style={'width': '30%', 'padding': '10px', 'border': '1px solid lightgray', 'borderRadius': '10px', 'margin': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    # Second Row: Line Plot
    html.Div([
        html.Div([
            html.H4('Price Trend (Line Plot)', style={'fontSize': 25, 'fontWeight': 'bold'}),
            dcc.Graph(figure=px.line(df, x='Year-Month', y='Price', markers=True)
                      .update_layout(xaxis_title="Time (Year-Month)", yaxis_title="Egg Price ($)"))
        ], style={'width': '100%', 'padding': '10px'}),
    ], style={'display': 'flex', 'marginTop': '20px'}),

    # Third row: Price Dist Histogram and Price Summary
    html.Div([
        html.Div([
            html.H4('Price Distribution (Histogram)', style={'fontSize': 25, 'fontWeight': 'bold'}),
            dcc.Graph(figure=px.histogram(df, x='Price', nbins=10))
        ], style={'width': '60%', 'padding': '10px'}),

        html.Div([
            html.H4('Price Summary', style={'fontSize': 25, 'fontWeight': 'bold'}),
            dash_table.DataTable(
                data=price_summary.to_dict('records'),
                columns=[{"name": i, "id": i} for i in price_summary.columns],
                page_size=6,            )
        ], style={'width': '35%', 'padding': '10px', 'minHeight': '300px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'}),

   # Fourth row: Tables (X, y, Total Records)
    html.Div([
        html.Div([
            html.H4('X Data', style={'textAlign': 'left', 'fontSize': 25, 'fontWeight': 'bold'}),
            dash_table.DataTable(
                data=X.to_dict('records'),
                columns=[{"name": i, "id": i} for i in X.columns],
                page_size=5,
                style_table={'overflowX': 'auto'}
            )
        ], style={'width': '36%', 'padding': '10px', 'fontSize': 15}),
        html.Div([
            html.H4('y Data', style={'textAlign': 'left', 'fontSize': 25, 'fontWeight': 'bold'}),
            dash_table.DataTable(
                data=y.to_dict('records'),
                columns=[{"name": i, "id": i} for i in y.columns],
                page_size=5,
            )
        ], style={'width': '20%', 'padding': '30px'}),
        html.Div([
            html.H5('Total Records', style={'textAlign': 'left', 'fontSize': 25, 'fontWeight': 'bold'}),
            html.Div(f"{len(df)}", style={
                'textAlign': 'center',
                'fontSize': 40,
                'color': 'purple',
                'padding-right': '30px',
                'maxHeight': '300px'})
                
        ], style={
            'width': '30%',
            'padding': '20px',
            'border': '1px solid lightgray',
            'borderRadius': '5px',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'flex-start'}),
    ], style={'display': 'flex'}),
    

    # Fifth row: Download buttons
    html.Div([
        html.Div([
            html.Button("Download X Data (CSV)", id="btn-download-x"),
            dcc.Download(id="download-x")
        ], style={'width': '30%', 'padding': '30px'}),
        html.Div([
            html.Button("Download y Data (CSV)", id="btn-download-y"),
            dcc.Download(id="download-y")
        ], style={'width': '30', 'padding': '30px'}),
    ], style={'display': 'flex'}),

    # Sixth row: Feature Importance
    html.Div([
        html.Div([
            html.H4('Feature Importance', style={'fontSize': 25, 'fontWeight': 'bold'}),
            dcc.Graph(
                figure=px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h'
                ).update_layout(
                    xaxis_title='Coefficient Value',
                    yaxis_title='Features',
                    showlegend=False
                )
            )
        ], style={'width': '100%', 'padding': '0px'}),
    ], style={'display': 'flex', 'marginTop': '20px'}),

    # Sixth row: Price Dist Histogram and Feature Correlation Heatmap
    html.Div([
        html.Div([
            html.H4('Feature Correlations (Heatmap)', style={'fontSize': 25, 'fontWeight': 'bold'}),
            dcc.Graph(
                figure=px.imshow(
                    corr_matrix,
                    text_auto=True,
                    height=700,
                    color_continuous_scale='Blues',
                    title='Correlation Matrix'
                ).update_layout(xaxis_title="Features", yaxis_title="Features")
            )
        ], style={'width': '100%', 'padding': '10px'}),
    ], style={'display': 'flex', 'marginTop': '20px'}),

    # Seventh row: Model Performances (R2 and RMSE)
    html.Div([
        html.H4('Model Performances', style={'textAlign': 'left', 'fontSize': 25, 'fontWeight': 'bold'}),
    ], style={'width': '100%', 'padding': '10px'}),

    html.Div([
        html.Div([
            dcc.Graph(figure=fig_pre)
        ], style={'width': '100%', 'padding': '10px'}),

        html.Div([
            dcc.Graph(
                figure=make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("R² Scores", "RMSE Scores"),
                    shared_xaxes=False,
                    horizontal_spacing=0.15
                )
                .add_trace(go.Bar(name='Linear R²',
                        x=["Top 3", "Top 4", "Top 5"],
                        y=[0.4681, 0.5887, 0.7403],
                        marker_color='cornflowerblue'), row=1, col=1)
                .add_trace(go.Bar(name='Random Forest R²',
                        x=["Top 3", "Top 4", "Top 5"],
                        y=[0.4856, 0.5468, 0.5468],
                        marker_color='royalblue'), row=1, col=1)
                .add_trace(go.Bar(name='Linear RMSE',
                        x=["Top 3", "Top 4", "Top 5"],
                        y=[0.5209, 0.4581, 0.3640],
                        marker_color='lightcoral'), row=1, col=2)
                .add_trace(go.Bar(name='Random Forest RMSE',
                        x=["Top 3", "Top 4", "Top 5"],
                        y=[0.5123, 0.4809, 0.4809],
                        marker_color='indianred'), row=1, col=2)
                .update_layout(
                    barmode='group',
                    title_text="R² and RMSE Scores by Model (Post Backtesting)",
                    height=450,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(t=60)
                )
                .update_xaxes(title_text="Top N Features", row=1, col=1)
                .update_xaxes(title_text="Top N Features", row=1, col=2)
                .update_yaxes(title_text="R² Score", row=1, col=1)
                .update_yaxes(title_text="RMSE", row=1, col=2)
            )
        ], style={'width': '100%', 'padding': '10px'}),
    ], style={'display': 'flex', 'marginTop': '30px'}),
], style={'backgroundColor': '#fffdf6ff', 'padding': '20px', 'marginTop': '30px'})

from dash.dependencies import Input, Output
import base64
import io

@app.callback(
    Output("download-x", "data"),
    Input("btn-download-x", "n_clicks"),
    prevent_initial_call=True
)
def download_x(n_clicks):
    if n_clicks is None:
        return None
    return dcc.send_data_frame(X.to_csv, "X.csv", index=False)

@app.callback(
    Output("download-y", "data"),
    Input("btn-download-y", "n_clicks"),
    prevent_initial_call=True
)
def download_y(n_clicks):
    if n_clicks is None:
        return None
    return dcc.send_data_frame(y.to_csv, "y.csv", index=False)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)