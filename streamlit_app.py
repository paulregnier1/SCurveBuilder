import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the web app
st.title("S-Curve Builder")

# Input parameters
shape = st.slider("Shape parameter", min_value=1, max_value=100, value=30)
max_cap = st.number_input("Max Capacity (in M)", value=40)
inversion_point = st.slider("Inflection Point", min_value=0.5, max_value=1.2, value=0.95)

# Loss Ratios 

loss_ratio = np.arange(0.4,1.4,0.01)

# Define the function to plot
def sigmoid_curve(x, shape, inversion_point, max_cap):
    return max_cap * (1 - (1 / (1 + np.exp(-shape * (x - inversion_point)))))

y_values = sigmoid_curve(loss_ratio, shape, inversion_point, max_cap)

# Plotting the curve
fig, ax = plt.subplots()
ax.plot(loss_ratio, y_values, label="S-Curve")
ax.set_xlabel('Loss Ratio (%)')
ax.set_ylabel('Capacity (in M)')
ax.set_title('S-Curve')

# Reverse the x-axis
ax.invert_xaxis()

# Show x-axis as percentages
ax.set_xticks(np.arange(0.4, 1.4, 0.15))
ax.set_xticklabels([f'{int(x*100)}%' for x in np.arange(0.4, 1.4, 0.15)])

ax.legend()

# Display the plot
st.pyplot(fig)
