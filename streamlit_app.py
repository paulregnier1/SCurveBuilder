import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the web app
st.title("S-Curve Builder")

# Input parameters
shape = st.slider("Shape parameter", min_value=1, max_value=70, value=30)
max_cap = st.number_input("Max Capacity (in M)", value=40)
inversion_point = st.slider("Inflection Point", min_value=0.5, max_value=1.2, value=0.95)

# Loss Ratios 

loss_ratio = np.arange(0.4,1.4,0.01)

# Define the function to plot
def sigmoid_curve(x, shape, inversion_point, max_cap):
    return max_cap * (1 - (1 / (1 + np.exp(-shape * (x - inversion_point)))))

y_values = sigmoid_curve(loss_ratio, shape, inversion_point, max_cap)

# Get user input for manual points
st.subheader("Add Custom Points")
manual_loss_ratios = st.text_input("Enter loss ratios (comma separated)", value="0.6, 1.0, 1.2")
manual_capacities = st.text_input("Enter capacities (comma separated)", value="10, 30, 35")

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
    # Plotting the curve
    fig, ax = plt.subplots()
    ax.plot(loss_ratio, y_values, label="S-Curve")
    
    # Plot manual points in red
    ax.scatter(manual_loss_ratios, manual_capacities, color='red', label="Manual Points")
    
    # Reverse the x-axis and show x-axis as percentages
    ax.invert_xaxis()
    ax.set_xticks(np.arange(0.5, 1.6, 0.1))
    ax.set_xticklabels([f'{int(x*100)}%' for x in np.arange(0.5, 1.6, 0.1)])
    
    ax.set_xlabel('Loss Ratio (%)')
    ax.set_ylabel('Capacity')
    ax.set_title('S-Curve with Manual Points')
    ax.legend()

    # Display the plot
    st.pyplot(fig)
