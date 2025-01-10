import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from playsound import playsound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import csv

# Dennis Kochmann - NN for metamaterials
# Regularisation? Vyhlazení?
# Omezení hodnot - constrains - do loos func ?
# Sledovat validační loos

# Global variables for input size, output size, hidden layers, and other parameters
input_size = 21 # 6
output_size = 6 # 21
hidden_layers = [64, 128, 256, 128, 64]
x_log_scale = False
y_log_scale = False
stop_training = False
model = None
loss_history = []  # Global variable for loss history
train_loss_history = []  # For training loss
val_loss_history = []    # For validation loss
test_loss_history = []    # For validation loss
num_epochs = 0     # Global variable for number of epochs
num_samples = 0
batch_size = 0
data_file_path = "" # Global variable for data file path
train_percent = 0.7 # Global variables for train,
val_percent = 0.15 # validation, and test split percentages
active_model_name = "New model"  # Default to 'New model'
criterion = None
means = []
stds = []
# log_columns = [6, 7, 8, 9, 10, 11, 12, 13, 17]  # Columns for logarithmic normalization
log_columns = [0, 1, 2, 3, 4, 5, 6, 7, 11]  # Columns for logarithmic normalization
standardize_columns = list(range(0, 21))  # Columns 6-27 for standardization

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(NeuralNet, self).__init__()
        self.hidden_layers = hidden_layers
        layers = []

        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(file_path):
    global means, stds
    data = []

    with open(file_path, 'r') as file:
        # Read first two rows for means and stds
        means_line = file.readline().strip().replace(":", "")
        stds_line = file.readline().strip().replace(":", "")
        
        # Extract means and stds starting from the 7th column
        means = list(map(float, means_line.split()[6:]))
        stds = list(map(float, stds_line.split()[6:]))
        
        # Read the rest of the data
        # Read the rest of the data
        for line in file:
            line = line.replace(":", "")
            values = list(map(float, line.split()))
            data.append(values)
    
    return np.array(data, dtype=np.float32)

# Initializing scale toggles
x_log_scale = False
y_log_scale = True

def toggle_x_log_scale():
    global x_log_scale
    x_log_scale = not x_log_scale
    update_plot()

def toggle_y_log_scale():
    global y_log_scale
    y_log_scale = not y_log_scale
    update_plot()

def hidden_layers_architecture(hidden_layers_str):
    try:
        # Split the input string by spaces
        layers = hidden_layers_str.split()
        
        # Ensure all parts are integers
        layers = [int(x) for x in layers]
        
        # Ensure each layer is positive and <= 8192
        for layer in layers:
            if layer <= 0 or layer > 8192:
                raise ValueError("Only positive integers up to 8192 allowed. (e.g. 64 128 128 64)")
        
        return layers
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Error: {str(e)}")
        return None  # Return None if there's an error

def update_plot_with_lr(current_learning_rate):
    """ Updates the plot with the current scale settings and actual learning rate """
    global train_loss_history, val_loss_history, test_loss_history, num_epochs, data_file_path, num_samples, batch_size  # Use global variables
    ax.clear()
    ax.plot(train_loss_history, label='Training Loss', color='orange')
    ax.plot(val_loss_history, label='Validation Loss', color='lightblue')
    ax.set_facecolor('#2b2b2b')  # Darker background for the axes
    ax.spines['bottom'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    ax.title.set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs. Epoch')
    ax.grid(True, color='red')

    # Define the text with the actual learning rate
    text1 = (f'Data:\n'
             f'Num of Epochs:\n'
             f'Learning Rate (actual):\n'
             f'Num of Layers:\n'
             f'Batch Size:\n'
             f'Samples:')
    
    # Update text2 to include the actual learning rate
    text2 = (f'{data_file_path}\n'
             f'{num_epochs:<}\n'
             f'{learning_rate:.2e} ({current_learning_rate:.2e})\n'  # Display the current learning rate
             f'{len(hidden_layers):<}\n'
             f'{batch_size:<}\n'
             f'{num_samples:<}')
    
    # Add description to plot with aligned text
    ax.text(0.68, 0.95, text1,
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.text(0.7, 0.95, text2,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    # Set axis scales based on toggles
    if x_log_scale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    if y_log_scale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')

    canvas.draw()


def update_plot():
    """ Updates the plot with the current scale settings """
    global train_loss_history, val_loss_history, test_loss_history, num_epochs, data_file_path, num_samples, batch_size  # Use global variables
    ax.clear()
    ax.plot(train_loss_history, label='Training Loss', color='orange')
    ax.plot(val_loss_history, label='Validation Loss', color='lightblue')
    ax.set_facecolor('#2b2b2b')  # Darker background for the axes
    ax.spines['bottom'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    ax.title.set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs. Epoch')
    ax.grid(True, color='red')

    # Define the text with alignment
    text1 = (f'Data:\n'
            f'Num of Epochs:\n'
            f'Learning Rate:\n'
            f'Num of Layers:\n'
            f'Batch Size:\n'
            f'Samples:')
    text2 = (f'{data_file_path}\n'
            f'{num_epochs:<}\n'
            f'{learning_rate:<}\n'
            f'{len(hidden_layers):<}\n'
            f'{batch_size:<}\n'
            f'{num_samples:<}')
    
    # Add the legend below the ax.text
    ax.legend(loc='upper right', fontsize=10, facecolor='red', edgecolor='white', bbox_to_anchor=(1.0, 0.6))


    # Add description to plot with aligned text
    ax.text(0.68, 0.95, text1,
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.text(0.7, 0.95, text2,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    # Set axis scales based on toggles
    if x_log_scale:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    if y_log_scale:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')

    canvas.draw()

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, ax, canvas, progress_label, time_label):
    global stop_training, train_loss_history, val_loss_history, num_epochs, data_file_path
    start_time = time.time()  # Record the start time

    # Define the StepLR scheduler
    step_size = 1024  # Adjust the learning rate every 10 epochs
    gamma = 1.0     # Reduce learning rate by a factor... 0.93
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        if stop_training:
            break
        
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Evaluate on validation data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_dataloader:
                val_outputs = model(val_X)
                val_loss_batch = criterion(val_outputs, val_y)
                val_loss += val_loss_batch.item() * val_X.size(0)
        
        epoch_val_loss = val_loss / len(val_dataloader.dataset)
        val_loss_history.append(epoch_val_loss)

        # Update the learning rate using StepLR scheduler
        scheduler.step()  # Call this after each epoch

        # Get the current learning rate
        current_learning_rate = optimizer.param_groups[0]['lr']

        # Update the plot and display the actual learning rate in the text
        update_plot_with_lr(current_learning_rate)

        progress_percentage = ((epoch + 1) / num_epochs) * 100
        progress_label.config(text=f'Epochs completed: {progress_percentage:.2f}%')

        # Calculate the elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        completed_epochs = epoch + 1
        remaining_epochs = num_epochs - completed_epochs
        if completed_epochs > 0:
            remaining_time = (elapsed_time / completed_epochs) * remaining_epochs
            minutes, seconds = divmod(remaining_time, 60)
            time_label.config(text=f'Estimated Remaining Time: {int(minutes):02d}:{int(seconds):02d}')

        root.update()

    # Play a sound after the training is completed
    try:
        playsound('/home/gb/Music/HoE3_sound01.mp3')  # Replace with the actual path to your sound file
    except Exception as e:
        messagebox.showerror("Error", f"Could not play sound: {str(e)}")


def make_predictions(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions.numpy()

def calculate_mean_percentage_error(true_values, predicted_values, split_index=3):
    # Initialize lists to collect valid errors
    first_values_error = []
    remaining_values_error = []

    # Calculate errors for the first `split_index` values
    for i in range(true_values.shape[0]):
        for j in range(split_index):
            if true_values[i, j] != 0:
                # Calculate the percentage error and add to the list
                error = np.abs((true_values[i, j] - predicted_values[i, j]) / true_values[i, j]) * 100
                first_values_error.append(error)

    # Calculate errors for the remaining values
    for i in range(true_values.shape[0]):
        for j in range(split_index, true_values.shape[1]):
            if true_values[i, j] != 0:
                # Calculate the percentage error and add to the list
                error = np.abs((true_values[i, j] - predicted_values[i, j]) / true_values[i, j]) * 100
                remaining_values_error.append(error)

    # Calculate the mean errors, ignoring zero-value cases
    mean_first_values_error = np.mean(first_values_error) if first_values_error else np.nan  # Handle empty list case
    mean_remaining_values_error = np.mean(remaining_values_error) if remaining_values_error else np.nan


    return mean_first_values_error, mean_remaining_values_error

def run_training():
    global stop_training, train_loss_history, val_loss_history, test_loss_history
    global model, optimizer, train_percent, val_percent, num_samples, batch_size, criterion
    global hidden_layers, active_model_name, num_epochs, data_file_path, learning_rate
    stop_training = False

    # Validate hidden layers input
    hidden_layers_str = hidden_layers_entry.get()  # Get the user's input for hidden layers
    validated_hidden_layers = hidden_layers_architecture(hidden_layers_str)

    if validated_hidden_layers is None:
        return  # Stop execution if the hidden layers input is invalid
    
    # Set the global hidden_layers variable
    hidden_layers = validated_hidden_layers

    # Check if a model already exists; if not, create a new one
    if model is None:
        # Create a new model if one isn't already loaded
        model = NeuralNet(input_size, output_size, hidden_layers)
        optimizer = optim.Adam(model.parameters(), lr=float(lr_entry.get()))  # Create new optimizer

        # Update the active model name to 'New model'
        active_model_name = "New model"
        active_model_label.config(text=f"Active model: {active_model_name}")

    criterion = nn.MSELoss()
    learning_rate = float(lr_entry.get())  # Ensure learning_rate is used correctly
    num_epochs = int(epochs_entry.get())

    data_file_path = f'{file_path_entry.get()}.txt'
    data = load_data(data_file_path)
    data = torch.tensor(data, dtype=torch.float32)

    # Shuffle data
    num_samples = data.shape[0]
    indices = torch.randperm(num_samples)
    data = data[indices]

     # Get percentages and batch from the GUI
    train_percent = float(train_percent_entry.get()) / 100  # Convert to decimal
    val_percent = float(val_percent_entry.get()) / 100      # Convert to decimal
    batch_size = int(batch_size_entry.get())  # Read batch size from user input


    # Split data using train_percent and val_percent
    train_size = int(train_percent * num_samples)
    val_size = int(val_percent * num_samples)
    test_size = num_samples - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Prepare input and output data
    y_train = train_data[:, :output_size]
    X_train = train_data[:, output_size:(input_size + output_size)]

    y_val = val_data[:, :output_size]
    X_val = val_data[:, output_size:(input_size + output_size)]

    y_test = test_data[:, :output_size]
    X_test = test_data[:, output_size:(input_size + output_size)]

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize loss histories
    train_loss_history.clear()
    val_loss_history.clear()
    test_loss_history.clear()

    # Train the model
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, ax, canvas, progress_label, time_label)

    # Evaluate on test data
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_X, test_y in test_dataloader:
            test_outputs = model(test_X)
            test_loss_batch = criterion(test_outputs, test_y)
            test_loss += test_loss_batch.item() * test_X.size(0)
    test_loss /= len(test_dataloader.dataset)
    test_loss_history.append(test_loss)
    print(f'Test Loss: {test_loss:.4f}')

    # Can also compute mean percentage errors on the test set if needed, Gandalf


def save_plot():
    num_epochs = int(epochs_entry.get())
    global learning_rate  # Ensure learning_rate is used from the global scope
    train_file_path = file_path_entry.get()
    
    # Add the number of epochs, learning rate, and data file path to the plot
    ax.text(0.95, 0.95, f'Num Epochs   : {num_epochs}\n'
                        f'Learning Rate: {learning_rate}\n'
                        f'Num of Layers: {len(hidden_layers)}\n'
                        f'Data : {train_file_path}',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    # Save the plot as a PNG file
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png")])
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("Save PNG", f"Plot saved as {file_path}")

def speed_save_plot():
    num_epochs = int(epochs_entry.get())
    global learning_rate, batch_size, hidden_layers  # Ensure learning_rate is used from the global scope
    
    # Add the number of epochs, learning rate, and data file path to the plot
    train_file_path = file_path_entry.get()
    
    # Get the training data file path to determine the save location
    dir_path = os.path.dirname(train_file_path)
    base_name = os.path.basename(train_file_path)
    current_time = time.strftime("%y%m%d-%H%M")
    save_path = os.path.join(dir_path, f"{base_name}_{len(hidden_layers)}-{batch_size}_{current_time}.png")
    
    fig.savefig(save_path)
    messagebox.showinfo("Speed Save PNG", f"Plot saved as {save_path}")

def save_losses_to_csv():
    global train_loss_history, val_loss_history, test_loss_history, data_file_path
    global num_samples, batch_size, hidden_layers, train_percent, val_percent
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    # Extract the last part of the file path (if exists)
    data_file_name = os.path.basename(data_file_path) if data_file_path else "N/A"
    
    if not file_path:
        return  # User cancelled the save dialog
    
    # Open the CSV file and write the loss data
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Test Loss',
                         'Samples', 'Batch Size', 'Layers Architecture',
                         'Train [%]', 'Validation [%]'])  # Write header
        
        # Find the number of epochs (train/val history length)
        num_epochs = len(train_loss_history)
        for epoch in range(num_epochs):
            # Write train and validation losses for each epoch
            writer.writerow([epoch + 1,  # Epoch number
                            train_loss_history[epoch],  # Training loss
                            val_loss_history[epoch],    # Validation loss
                            test_loss_history[0] if epoch == 0 else "",
                            num_samples if epoch == 0 else "",
                            batch_size if epoch  == 0 else "",
                            hidden_layers[epoch] if epoch < len(hidden_layers) else "",
                            train_percent if epoch == 0 else "",
                            val_percent if epoch == 0 else ""])

    messagebox.showinfo("CSV Saved", f"Losses have been saved to {file_path}")

def speed_save_losses_to_csv():
    global train_loss_history, val_loss_history, test_loss_history, data_file_path
    global num_samples, batch_size, hidden_layers, train_percent, val_percent
    # # Ask the user where to save the file
    # file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    # Extract the last part of the file path (if exists)
    # data_file_name = os.path.basename(data_file_path) if data_file_path else "N/A"
    
    # if not file_path:
    #     return  # User cancelled the save dialog

    # Get the training data file path to determine the save location
    train_file_path = file_path_entry.get()
    dir_path = os.path.dirname(train_file_path)
    base_name = os.path.basename(train_file_path)
    current_time = time.strftime("%y%m%d-%H%M")
    save_path = os.path.join(dir_path, f"{base_name}_{len(hidden_layers)}-{batch_size}_{current_time}.csv")
    
    # Open the CSV file and write the loss data
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Test Loss',
                         'Samples', 'Batch Size', 'Layers Architecture',
                         'Train [%]', 'Validation [%]'])  # Write header
        
        # Find the number of epochs (train/val history length)
        num_epochs = len(train_loss_history)
        for epoch in range(num_epochs):
            # Write train and validation losses for each epoch
            writer.writerow([epoch + 1,  # Epoch number
                            train_loss_history[epoch],  # Training loss
                            val_loss_history[epoch],    # Validation loss
                            test_loss_history[0] if epoch == 0 else "",
                            num_samples if epoch == 0 else "",
                            batch_size if epoch  == 0 else "",
                            hidden_layers[epoch] if epoch < len(hidden_layers) else "",
                            train_percent if epoch == 0 else "",
                            val_percent if epoch == 0 else ""])

    messagebox.showinfo("CSV Saved", f"Losses have been saved to {save_path}")



def shutdown():
    global stop_training
    stop_training = True

def save_model():
    global active_model_name
    file_path = filedialog.asksaveasfilename(defaultextension=".pth",
                                             filetypes=[("PyTorch model files", "*.pth")])
    if file_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': int(epochs_entry.get()),  # Saving current epoch count
            'batch_size': batch_size,  # Saving batch size
            'hidden_layers': hidden_layers,  # Saving hidden layers architecture
            'learning_rate': learning_rate,  # Saving learning rate
            'input_size': input_size,  # Save input size
            'output_size': output_size  # Save output size
        }, file_path)

        # Update the active model name
        active_model_name = os.path.basename(file_path)  # Extract the file name without the path
        active_model_label.config(text=f"Active model: {active_model_name}")

        messagebox.showinfo("Save Model", f"Model saved as {file_path}")

def load_model():
    global model, optimizer, batch_size, hidden_layers
    global active_model_name, learning_rate, input_size, output_size

    file_path = filedialog.askopenfilename(filetypes=[("PyTorch model files", "*.pth")])
    if file_path:
        # Load the checkpoint first to get the architecture and parameters
        checkpoint = torch.load(file_path)

        # Load additional parameters (batch size, hidden layers, learning rate, input/output size)
        batch_size = checkpoint.get('batch_size', batch_size)
        hidden_layers = checkpoint.get('hidden_layers', hidden_layers)
        learning_rate = checkpoint.get('learning_rate', learning_rate)
        input_size = checkpoint.get('input_size', input_size)
        output_size = checkpoint.get('output_size', output_size)

        # Reinitialize the model with the loaded architecture
        model = NeuralNet(input_size, output_size, hidden_layers)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Load the saved model state and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Set model to evaluation mode if needed
        model.eval()

        # Update the corresponding GUI entries (optional)
        active_model_name = os.path.basename(file_path)  # Extract the file name without the path
        active_model_label.config(text=f"Active model: {active_model_name}")

        batch_size_entry.delete(0, tk.END)
        batch_size_entry.insert(0, str(batch_size))

        hidden_layers_entry.delete(0, tk.END)
        hidden_layers_entry.insert(0, " ".join(map(str, hidden_layers)))

        lr_entry.delete(0, tk.END)
        lr_entry.insert(0, str(learning_rate))

        messagebox.showinfo("Load Model", f"Model loaded from {file_path}")

def make_custom_prediction(input_fields):
    global model
    
    # Collect input values from the user
    user_input = []
    for field in input_fields:
        try:
            value = float(field.get())
            user_input.append(value)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

    # Ensure the input has exactly 21 values
    if len(user_input) != input_size:
        messagebox.showerror("Invalid Input", f"Please enter exactly {input_size} values.")
        return

    # Standardize the user input (includes log normalization for specific columns)
    standardized_input = standardize_input(user_input)

    # Convert the input into a torch tensor
    user_input_tensor = torch.tensor(standardized_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Make a prediction using the trained model
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        output = model(user_input_tensor)  # Get the predicted values
        softmax = nn.Softmax(dim=1)  # Apply softmax to get probabilities
        probabilities = softmax(output)

    # Convert tensors to numpy arrays for easy handling
    predicted_values = output.numpy()[0]
    predicted_probabilities = probabilities.numpy()[0]

    # Display the predicted values and probabilities
    prediction_text = f"Predicted Values: {predicted_values}\nProbabilities: {predicted_probabilities}"
    prediction_label.config(text=prediction_text)

def standardize_input(user_input):
    global means, stds
    standardized_input = []

    for i, (x, mean, std) in enumerate(zip(user_input, means, stds)):
        # Apply log transformation for log_columns
        if i in log_columns and x > 0:
            x = np.log(x)
        
        # Standardize the input
        standardized_value = (x - mean) / std if std != 0 else x
        standardized_input.append(standardized_value)
    
    return standardized_input

# GUI Setup
root = tk.Tk()
root.title("PyTorch Neural Network Training")
root.configure(bg='#2b2b2b')

tk.Label(root, text="Training Data File Path (prefix):", bg='#2b2b2b', fg='white').grid(row=0, column=0)
file_path_entry = tk.Entry(root, width=50, bg='grey', fg='white')
file_path_entry.grid(row=0, column=1)
file_path_entry.insert(0, "data/dataStandard01")

tk.Label(root, text="Layers Architecture:", bg='#2b2b2b', fg='white').grid(row=1, column=0)
hidden_layers_entry = tk.Entry(root, width=20, bg='grey', fg='white')
hidden_layers_entry.grid(row=1, column=1)
hidden_layers_entry.insert(2, "64 128 256 512 256 128 64 32")  # Default hidden layers

tk.Label(root, text="Number of Epochs:", bg='#2b2b2b', fg='white').grid(row=2, column=0)
epochs_entry = tk.Entry(root, width=10, bg='grey', fg='white')
epochs_entry.grid(row=2, column=1)
epochs_entry.insert(0, "2048")

tk.Label(root, text="Learning Rate:", bg='#2b2b2b', fg='white').grid(row=3, column=0)
lr_entry = tk.Entry(root, width=10, bg='grey', fg='white')
lr_entry.grid(row=3, column=1)
lr_entry.insert(0, "1e-4")

tk.Label(root, text="Training Percentage:", bg='#2b2b2b', fg='white').grid(row=4, column=0)
train_percent_entry = tk.Entry(root, width=10, bg='grey', fg='white')
train_percent_entry.grid(row=4, column=1)
train_percent_entry.insert(0, "70")  # Default is 70%

tk.Label(root, text="Validation Percentage:", bg='#2b2b2b', fg='white').grid(row=5, column=0)
val_percent_entry = tk.Entry(root, width=10, bg='grey', fg='white')
val_percent_entry.grid(row=5, column=1)
val_percent_entry.insert(0, "15")  # Default is 15%

tk.Label(root, text="Batch Size:", bg='#2b2b2b', fg='white').grid(row=6, column=0)
batch_size_entry = tk.Entry(root, width=10, bg='grey', fg='white')
batch_size_entry.grid(row=6, column=1)
batch_size_entry.insert(0, "32")  # Default batch size is set to 32

train_button = tk.Button(root, text="Train", command=run_training, bg='grey', fg='black')
train_button.grid(row=8, column=0, columnspan=2)

y_scale_button = tk.Button(root, text="Toggle Y Log Scale", command=toggle_y_log_scale, bg='grey', fg='black')
y_scale_button.grid(row=9, column=2)

save_losses_button = tk.Button(root, text="Save as CSV", command=save_losses_to_csv, bg='grey', fg='black')
save_losses_button.grid(row=10, column=2, columnspan=2)

speed_save_losses_button = tk.Button(root, text="Save CSV", command=speed_save_losses_to_csv, bg='grey', fg='black')
speed_save_losses_button.grid(row=11, column=2, columnspan=2)

save_button = tk.Button(root, text="Save AS PNG", command=save_plot, bg='grey', fg='black')
save_button.grid(row=10, column=3, columnspan=3)

speed_save_button = tk.Button(root, text="Save PNG", command=speed_save_plot, bg='grey', fg='black')
speed_save_button.grid(row=11, column=3, columnspan=3)

progress_label = tk.Label(root, text="", bg='#2b2b2b', fg='white')
progress_label.grid(row=10, column=0, columnspan=2)

time_label = tk.Label(root, text="", bg='#2b2b2b', fg='white')  # Label to show remaining time
time_label.grid(row=11, column=0, columnspan=2)

save_model_button = tk.Button(root, text="Save Model", command=save_model, bg='grey', fg='black')
save_model_button.grid(row=12, column=0, columnspan=2)

load_model_button = tk.Button(root, text="Load Model", command=load_model, bg='grey', fg='black')
load_model_button.grid(row=13, column=0, columnspan=2)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=14, column=0, columnspan=2)
fig.patch.set_facecolor('#2b2b2b')  # Set the background color of the figure

shutdown_button = tk.Button(root, text="Shutdown", command=shutdown, bg='red', fg='black')
shutdown_button.grid(row=15, column=0, columnspan=2)

active_model_label = tk.Label(root, text=f"Active model: {active_model_name}", bg='#2b2b2b', fg='white')
active_model_label.grid(row=16, column=0, columnspan=2)

# Define the positions of the inputs in the triangular matrix (6x6)
positions = [
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),  # Diagonal elements
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),          # Second column
    (0, 2), (1, 3), (2, 4), (3, 5),                  # Third column
    (0, 3), (1, 4), (2, 5),                          # Fourth column
    (0, 4), (1, 5),                                  # Fifth column
    (0, 5)                                           # Last element
]

# List of default values to pre-fill in the input fields
# default_values_21 = [1.1955, 1.2036, 1.2160, 1.6084, 1.4958, 1.6755, 1.1242, 1.0393, -0.3395, -0.6866, -0.7960, 1.0982, -1.0097, -0.4790, -0.7523, -0.7062, -0.7772, -0.2954, -1.1331, -0.8369, 0.1445
#     ]
default_values_21 = [93.9697,  93.2161,  91.7952, 142.4630, 123.4954, 137.1729,  78.8051,  76.4141,  -7.1687,  -1.3738,   1.5308,  77.6191,  13.5696,  -3.6482,  -0.3952,   3.5051,   1.0654, -38.5380,   4.6255, -36.9746, -37.3557
    ]

# Create input fields for 21 values in triangular matrix format
input_fields = []
for idx, (row, col) in enumerate(positions):
    label = tk.Label(root, text=f"Input {idx + 1}:", bg='#2b2b2b', fg='white')
    label.grid(row=row, column=col * 2 + 2)  # Place label
    entry = tk.Entry(root, width=10, bg='grey', fg='white')

    # Insert the default value from the list for each input field
    entry.insert(0, default_values_21[idx])

    entry.grid(row=row, column=col * 2 + 3)  # Place entry field next to the label
    input_fields.append(entry)

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=lambda: make_custom_prediction(input_fields), bg='grey', fg='black')
predict_button.grid(row=12, column=2, columnspan=2)

# Label to show predicted values, probabilities, and loss
prediction_label = tk.Label(root, text="", bg='#2b2b2b', fg='white')
prediction_label.grid(row=13, column=2, columnspan=2)
# # # probability_label = tk.Label(root, text="", bg='#2b2b2b', fg='white')
# # # probability_label.grid(row=15, column=2, columnspan=2)
loss_label = tk.Label(root, text="", bg='#2b2b2b', fg='white')
loss_label.grid(row=14, column=2, columnspan=2)

# Label to display input size
input_size_label = tk.Label(root, text=f"Input Size: {input_size}", bg='#2b2b2b', fg='white')
input_size_label.grid(row=15, column=2, columnspan=2)

# Label to display output size
output_size_label = tk.Label(root, text=f"Output Size: {output_size}", bg='#2b2b2b', fg='white')
output_size_label.grid(row=16, column=2, columnspan=2)

stop_training = False
model = None

root.mainloop()