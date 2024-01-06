import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Loading the pre-trained model for Sign Language Detection
sign_language_model = load_model('sign_1.h5')

# GUI setup
top = tk.Tk()
top.geometry("800x600")
top.title("Sign Language Detector")
top.configure(background='#CDCDCD')

# Labels and image display setup
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    global label1
    # Preprocess image for prediction
    image = Image.open(file_path)
    image = image.resize((150, 150))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0

    # Make prediction using the model
    prediction = sign_language_model.predict(image)
    class_index = np.argmax(prediction)
    # Mapping class index to alphabets (adjust this mapping based on your data)
    class_index_to_label = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',36: 'Space'
}

    
    # Display prediction result
    predicted_alphabet = class_index_to_label.get(class_index, 'Unknown')
    label1.configure(foreground='#011638', text="Predicted Sign: " + predicted_alphabet)
    #label1.configure(foreground='#011638', text="Predicted Sign: " + str(class_index))

def show_detect_button(file_path):
    Detect_b = Button(top, text="Make Prediction", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background='#364156', foreground='White', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        # Display selected image
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text="")
        show_detect_button(file_path)
    except Exception as e:
        print(e)

# Upload button setup
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='White', font=('arial', 10, 'bold'))

upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)

# Packing labels and heading
label1.pack(side='bottom', pady=50)
heading = Label(top, text="Sign Language Detector", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground='#364156')
heading.pack()

# Main loop for GUI
top.mainloop()
