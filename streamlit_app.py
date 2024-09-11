import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Title of the web app
st.title("S-Curve Builder")

# Input parameters
shape = st.slider("Shape parameter", min_value=1, max_value=70, value=30)
max_cap = st.number_input("Max Capacity (in M)", value=40)
inversion_point = st.slider("Inflection Point", min_value=0.5, max_value=1.2, value=0.95)

# Loss Ratios 
loss_ratio = np.arange(0.4, 1.4, 0.01)

# Define the function to plot
def sigmoid_curve(x, shape, inversion_point, max_cap):
    return max_cap * (1 - (1 / (1 + np.exp(-shape * (x - inversion_point)))))

y_values = sigmoid_curve(loss_ratio, shape, inversion_point, max_cap)

# Get user input for manual points
st.subheader("Add Custom Points")
manual_loss_ratios = st.text_input("Enter loss ratios (comma separated)", value="0.5,0.8,0.85,0.9,1,1.2,1.25")
manual_capacities = st.text_input("Enter capacities (comma separated)", value="40,32,28,15,8,6,4")

# Convert the inputs into lists of floats
try:
    manual_loss_ratios = [float(x.strip()) for x in manual_loss_ratios.split(",")]
    manual_capacities = [float(x.strip()) for x in manual_capacities.split(",")]
except ValueError:
    st.error("Please enter valid numbers for loss ratios and capacities.")

# Ensure that the number of loss ratios and capacities match
if len(manual_loss_ratios) != len(manual_capacities):
    st.error("Number of loss ratios must match number of capacities.")
else:
    # Fit the S-curve to the manual points
    popt, _ = curve_fit(sigmoid_curve, manual_loss_ratios, manual_capacities, 
                        p0=[shape, inversion_point, max_cap], 
                        bounds=([1, 0.5, 0], [70, 1.2, np.inf]))
    
    fitted_shape, fitted_inversion, fitted_max_cap = popt
    fitted_y_values = sigmoid_curve(loss_ratio, *popt)

    # Toggle button for fitted curve
    show_fitted_curve = st.checkbox("Show fitted S-curve", value=True)

    # Plotting the curve
    fig, ax = plt.subplots()
    ax.plot(loss_ratio, y_values, label="Original S-Curve")
    
    # Plot manual points in red
    ax.scatter(manual_loss_ratios, manual_capacities, color='red', label="Manual Points")
    
    # Plot fitted curve if toggle is on
    if show_fitted_curve:
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
    if show_fitted_curve:
        st.subheader("Fitted S-Curve Parameters")
        st.write(f"Shape: {fitted_shape:.2f}")
        st.write(f"Inversion Point: {fitted_inversion:.2f}")
        st.write(f"Max Capacity: {fitted_max_cap:.2f}")
    # Display the plot
    st.pyplot(fig)
