import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Set seaborn style
sns.set(style="whitegrid")

# Number of experiments and epochs
exp_nums = 10
epochs = 20  # Assuming each file contains 20 epochs of data

# Generate colors for experiments
palette = sns.color_palette("tab10", exp_nums + 1)

def read_loss_file(filepath):
    with open(filepath, "r") as f:
        return ast.literal_eval(f.read().strip())  # Parse as a Python list

# Create plot
plt.figure(figsize=(10, 6))

for exp_num in range(1, exp_nums + 1):
    # Load data
    train_loss = read_loss_file(f"results/exp{exp_num}_train_loss.txt")
    val_loss = read_loss_file(f"results/exp{exp_num}_val_loss.txt")

    # Ensure the x-axis reflects epochs 1 to 20
    epochs_range = np.arange(1, epochs + 1)

    # Plot training loss (solid line)
    plt.plot(epochs_range, train_loss, linestyle="-", color=palette[exp_num],
             label=f"Exp {exp_num} Train")
    
    # Plot validation loss (dashed line)
    plt.plot(epochs_range, val_loss, linestyle="--", color=palette[exp_num],
             label=f"Exp {exp_num} Val")

# Add labels and title
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Loss per Experiment", fontsize=14)

# Adjust x-axis ticks and limits to show exactly 1 to 20
plt.xticks(epochs_range)
# plt.xlim(0.9, epochs)

# Place legend outside of the plot and use a smaller font size
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

plt.tight_layout()
plt.savefig("loss.png", bbox_inches="tight")
plt.show()
