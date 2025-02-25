import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Define folder
results_folder = "results"

# Get all loss files
train_files = sorted(glob.glob(os.path.join(results_folder, "exp*_train_loss.txt")))
val_files = sorted(glob.glob(os.path.join(results_folder, "exp*_val_loss.txt")))

# Extract experiment numbers
exp_nums = sorted(set(int(f.split("_")[0][3:]) for f in train_files))

# Generate colors for experiments
palette = sns.color_palette("tab10", len(exp_nums))

# Create plot
plt.figure(figsize=(10, 6))

for i, exp_num in enumerate(exp_nums):
    # Load data
    train_loss = np.loadtxt(os.path.join(results_folder, f"exp{exp_num}_train_loss.txt"))
    val_loss = np.loadtxt(os.path.join(results_folder, f"exp{exp_num}_val_loss.txt"))
    
    # Plot training loss (solid line)
    plt.plot(train_loss, linestyle="-", color=palette[i], label=f"Exp {exp_num} Train")
    
    # Plot validation loss (dashed line)
    plt.plot(val_loss, linestyle="--", color=palette[i], label=f"Exp {exp_num} Val")

# Add labels and legend
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Experiment")
plt.legend()
plt.show()
