import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pandas as pd
import io
import base64

st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stSidebar {
    background-color: #e0e2e6;
}
</style>
""", unsafe_allow_html=True)

# Title of the web app
st.title("S-Curve Builder")

# Sidebar for input parameters
st.sidebar.header("S-Curve Parameters")

with st.sidebar.form("parameters_form"):
    shape = st.slider("Shape parameter", min_value=1, max_value=70, value=30)
    max_cap = st.number_input("Max Capacity (in M)", value=40)
    inversion_point = st.slider("Inflection Point", min_value=0.5, max_value=1.2, value=0.95)
    
    num_points = st.number_input("Number of custom points", min_value=1, max_value=10, value=5)
    
    manual_loss_ratios = []
    manual_capacities = []
    
    for i in range(num_points):
        col1, col2 = st.columns(2)
        with col1:
            loss_ratio_point = st.number_input(f"Loss Ratio {i+1}", min_value=0.4, max_value=1.4, value=0.8, step=0.01, key=f"loss_ratio_{i}")
        with col2:
            capacity_point = st.number_input(f"Capacity {i+1}", min_value=0.0, max_value=float(max_cap), value=20.0, step=0.1, key=f"capacity_{i}")
        manual_loss_ratios.append(loss_ratio_point)
        manual_capacities.append(capacity_point)
    
    update_button = st.form_submit_button("Update Plot")

# Loss Ratios 
loss_ratio = np.arange(0.4, 1.4, 0.01)

# Define the function to plot
def sigmoid_curve(x, shape, inversion_point, max_cap):
    return max_cap * (1 - (1 / (1 + np.exp(-shape * (x - inversion_point)))))

y_values = sigmoid_curve(loss_ratio, shape, inversion_point, max_cap)

# Fit the S-curve to the manual points
try:
    popt, pcov = curve_fit(sigmoid_curve, manual_loss_ratios, manual_capacities, 
                           p0=[shape, inversion_point, max_cap], 
                           bounds=([1, 0.5, 0], [70, 1.2, np.inf]))
    fitted_shape, fitted_inversion, fitted_max_cap = popt
    fitted_y_values = sigmoid_curve(loss_ratio, *popt)
    fit_success = True
    
    # Calculate error metrics
    residuals = np.array(manual_capacities) - sigmoid_curve(np.array(manual_loss_ratios), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(manual_capacities) - np.mean(manual_capacities))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    
except:
    st.warning("Unable to fit curve to the provided points. Try adjusting your points or parameters.")
    fit_success = False

# Create Plotly figure
fig = go.Figure()

# Add original S-curve
fig.add_trace(go.Scatter(x=loss_ratio, y=y_values, mode='lines', name='Original S-Curve'))

# Add manual points
fig.add_trace(go.Scatter(x=manual_loss_ratios, y=manual_capacities, mode='markers', name='Manual Points', marker=dict(color='red', size=10)))

# Add fitted curve if successful
if fit_success:
    fig.add_trace(go.Scatter(x=loss_ratio, y=fitted_y_values, mode='lines', name='Fitted S-Curve', line=dict(dash='dash', color='green')))

# Update layout
fig.update_layout(
    xaxis_title='Loss Ratio (%)',
    yaxis_title='Capacity',
    title='S-Curve with Manual Points and Fitted Curve',
    xaxis=dict(tickformat=',.0%', autorange="reversed"),
    hovermode='closest'
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Display fitted parameters and error metrics
if fit_success:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fitted S-Curve Parameters")
        st.write(f"Shape: {fitted_shape:.2f}")
        st.write(f"Inversion Point: {fitted_inversion:.2f}")
        st.write(f"Max Capacity: {fitted_max_cap:.2f}")
    with col2:
        st.subheader("Error Metrics")
        st.write(f"R-squared: {r_squared:.4f}")
        st.write(f"RMSE: {rmse:.4f}")

# Download options
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="s_curve_data.csv">Download CSV File</a>'
    return href

if st.button('Prepare Download'):
    df = pd.DataFrame({
        'Loss Ratio': loss_ratio,
        'Original S-Curve': y_values,
        'Fitted S-Curve': fitted_y_values if fit_success else np.nan
    })
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)
