import tkinter as tk
from tkinter import filedialog
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import threading

# Function to update the processing label
def update_processing_label(text):
    processing_label.config(text=text)
    root.update_idletasks()

# Function to process images in a separate thread
def process_images_thread():
    def process_single_image(path):
        base = os.path.splitext(os.path.basename(path))[0]
        print(base)

        # Update the processing label
        update_processing_label(f"Processing: {base}")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output_path = os.path.join(output_folder.get(), f'{base}_rlt.png')
        cv2.imwrite(output_path, output)

    model_path = 'models/esrganx4.pth'  # Specify your model path
    device = torch.device('cpu')  # Use 'cpu' if you want to run on CPU

    test_img_folder = input_folder.get()

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    for idx, path in enumerate(glob.glob(os.path.join(test_img_folder, '*.png'))):
        process_single_image(path)
    for idx, path in enumerate(glob.glob(os.path.join(test_img_folder, '*.jpg'))):
        process_single_image(path)

    # All files processed, close the GUI
    root.quit()

# Function to process images when the button is clicked
def process_images():
    processing_thread = threading.Thread(target=process_images_thread)
    processing_thread.start()

# Create the main window
root = tk.Tk()
root.title("ESRGAN Image Processing")

# Create and set variables for input and output folders
input_folder = tk.StringVar()
output_folder = tk.StringVar()

# Function to browse for input folder
def browse_input_folder():
    folder = filedialog.askdirectory()
    if folder:
        input_folder.set(folder)

# Function to browse for output folder
def browse_output_folder():
    folder = filedialog.askdirectory()
    if folder:
        output_folder.set(folder)

# Create labels and buttons
label_input = tk.Label(root, text="Select Input Folder:")
entry_input = tk.Entry(root, textvariable=input_folder, state="readonly")
button_browse_input = tk.Button(root, text="Browse", command=browse_input_folder)

label_output = tk.Label(root, text="Select Output Folder:")
entry_output = tk.Entry(root, textvariable=output_folder, state="readonly")
button_browse_output = tk.Button(root, text="Browse", command=browse_output_folder)

processing_label = tk.Label(root, text="Processing: ")
button_process = tk.Button(root, text="Process Images", command=process_images)

# Pack the widgets into the window
label_input.pack()
entry_input.pack()
button_browse_input.pack()
label_output.pack()
entry_output.pack()
button_browse_output.pack()
processing_label.pack()
button_process.pack()

# Start the GUI main loop
root.mainloop()
