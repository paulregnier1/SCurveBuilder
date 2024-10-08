import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Title of the web app
st.title("S-Curve Builder")

# Sidebar for input parameters
st.sidebar.header("S-Curve Parameters")
shape = st.sidebar.slider("Shape parameter", min_value=1, max_value=70, value=30)
max_cap = st.sidebar.number_input("Max Capacity (in M)", value=40)
inversion_point = st.sidebar.slider("Inflection Point", min_value=0.5, max_value=1.2, value=0.95)

# Loss Ratios 
loss_ratio = np.arange(0.4, 1.4, 0.01)

# Define the function to plot
def sigmoid_curve(x, shape, inversion_point, max_cap):
    return max_cap * (1 - (1 / (1 + np.exp(-shape * (x - inversion_point)))))

y_values = sigmoid_curve(loss_ratio, shape, inversion_point, max_cap)

# Manual point input in sidebar
st.sidebar.subheader("Add Custom Points")
num_points = st.sidebar.number_input("Number of custom points", min_value=1, max_value=10, value=1)

manual_loss_ratios = []
manual_capacities = []

for i in range(num_points):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        loss_ratio_point = col1.number_input(f"Loss Ratio {i+1}", min_value=0.4, max_value=1.4, value=0.8, step=0.01, key=f"loss_ratio_{i}")
    with col2:
        capacity_point = col2.number_input(f"Capacity {i+1}", min_value=0.0, max_value=float(max_cap), value=20.0, step=0.1, key=f"capacity_{i}")
    manual_loss_ratios.append(loss_ratio_point)
    manual_capacities.append(capacity_point)

# Fit the S-curve to the manual points
try:
    popt, _ = curve_fit(sigmoid_curve, manual_loss_ratios, manual_capacities, 
                        p0=[shape, inversion_point, max_cap], 
                        bounds=([1, 0.5, 0], [70, 1.2, np.inf]))
    fitted_shape, fitted_inversion, fitted_max_cap = popt
    fitted_y_values = sigmoid_curve(loss_ratio, *popt)
    fit_success = True
except:
    st.warning("Unable to fit curve to the provided points. Try adjusting your points or parameters.")
    fit_success = False

# Toggle button for fitted curve
show_fitted_curve = st.sidebar.checkbox("Show fitted S-curve", value=True)

# Plotting the curve
fig, ax = plt.subplots()
ax.plot(loss_ratio, y_values, label="Original S-Curve")

# Plot manual points in red
ax.scatter(manual_loss_ratios, manual_capacities, color='red', label="Manual Points")

# Plot fitted curve if toggle is on and fit was successful
if show_fitted_curve and fit_success:
    ax.plot(loss_ratio, fitted_y_values, '--', color='green', label="Fitted S-Curve")

# Reverse the x-axis and show x-axis as percentages
ax.invert_xaxis()
ax.set_xticks(np.arange(0.5, 1.6, 0.1))
ax.set_xticklabels([f'{int(x*100)}%' for x in np.arange(0.5, 1.6, 0.1)])

ax.set_xlabel('Loss Ratio (%)')
ax.set_ylabel('Capacity')
ax.set_title('S-Curve with Manual Points and Fitted Curve')
ax.legend()

# Display the plot
st.pyplot(fig)

# Display fitted parameters
if show_fitted_curve and fit_success:
    st.sidebar.subheader("Fitted S-Curve Parameters")
    st.sidebar.write(f"Shape: {fitted_shape:.2f}")
    st.sidebar.write(f"Inversion Point: {fitted_inversion:.2f}")
    st.sidebar.write(f"Max Capacity: {fitted_max_cap:.2f}")
