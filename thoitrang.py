import tkinter as tk
import pandas as pd
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Đường dẫn đến các tài nguyên
MODEL_PATH = "C:/Users/HoaMi/Downloads/fashion_model1.h5"
SCALER_PATH = "C:/Users/HoaMi/Downloads/scaler1.pkl"
LABEL_ENCODER_PATH = "C:/Users/HoaMi/Downloads/label_encoder1.pkl"

# Các tệp ảnh cho mỗi loại trang phục
IMAGE_MAP = {
    'Váy mini': 'C:/Users/HoaMi/Downloads/App human/static/images/vaymini.jpg',
    'Quần jeans skinny': 'C:/Users/HoaMi/Downloads/App human/static/images/jeansskinny.jpg',
    'Quần short': 'C:/Users/HoaMi/Downloads/App human/static/images/shorts.jpg',
    'Quần legging': 'C:/Users/HoaMi/Downloads/App human/static/images/leggings.jpg',
    'Quần ống loe': 'C:/Users/HoaMi/Downloads/App human/static/images/ongloe.jpg',
    'Quần jogger': 'C:/Users/HoaMi/Downloads/App human/static/images/joggers.jpg',
    'Quần suông': 'C:/Users/HoaMi/Downloads/App human/static/images/suông.jpg' , 
    'Quần kaki': 'C:/Users/HoaMi/Downloads/App human/static/images/kaki.jpg',
    'Đầm suông': 'C:/Users/HoaMi/Downloads/App human/static/images/damsuong.jpg', 
    'Quần ống rộng': 'C:/Users/HoaMi/Downloads/App human/static/images/ongrong.jpg',
    'Áo blouse lụa': 'C:/Users/HoaMi/Downloads/App human/static/images/blouselua.jpg', 
    'Áo crop top': 'C:/Users/HoaMi/Downloads/App human/static/images/croptop.jpg',
    'Áo phông ôm': 'C:/Users/HoaMi/Downloads/App human/static/images/aophongom.jpg', 
    'Áo tank top': 'C:/Users/HoaMi/Downloads/App human/static/images/tanktop.jpg',
    'Áo sơ mi': 'C:/Users/HoaMi/Downloads/App human/static/images/aosomi.png', 
    'Áo len mỏng': "C:/Users/HoaMi/Downloads/App human/static/images/aolenmong.jpg",
    'Blazer': 'C:/Users/HoaMi/Downloads/App human/static/images/blazer.jpg', 
    'Áo khoác da': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacda.jpg',
    'Áo khoác dày': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacday.jpg', 
    'Áo măng tô': 'C:/Users/HoaMi/Downloads/App human/static/images/mangto.jpg',
    'Áo hai dây': 'C:/Users/HoaMi/Downloads/App human/static/images/aohaiday.jpg', 
    'Áo yếm': 'C:/Users/HoaMi/Downloads/App human/static/images/aoyem.jpg',
    'Áo không tay': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhongtay.jpg',
    'Áo len': 'C:/Users/HoaMi/Downloads/App human/static/images/aolen.jpg', 
    'Áo thun dài tay': 'C:/Users/HoaMi/Downloads/App human/static/images/aothundaitay.jpg',
    'Hoodie': 'C:/Users/HoaMi/Downloads/App human/static/images/hoodie.jpg', 
    'Áo nỉ có mũ': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacni.jpg',
    'Áo parka': 'C:/Users/HoaMi/Downloads/App human/static/images/aoparka.jpg', 
    'Áo khoác ngoài': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacngoai.jpg',
    'Đầm xòe': 'C:/Users/HoaMi/Downloads/App human/static/images/damxoe.jpg', 
    'Đầm chữ A': 'C:/Users/HoaMi/Downloads/App human/static/images/damchua.jpg',
    'Đầm bao': 'C:/Users/HoaMi/Downloads/App human/static/images/dambao.jpg',
    'Đầm ôm': 'C:/Users/HoaMi/Downloads/App human/static/images/damom.jpg',
    'Chân váy bút chì': 'C:/Users/HoaMi/Downloads/App human/static/images/chanvaybutchi.jpg', 
    'Chân váy midi': 'C:/Users/HoaMi/Downloads/App human/static/images/chanvaymidi.jpg',
    'Đầm maxi': 'C:/Users/HoaMi/Downloads/App human/static/images/dammaxi.jpg', 
    'Đầm dạ hội': 'C:/Users/HoaMi/Downloads/App human/static/images/damdahoi.jpg',
    'Quần jumpsuit': 'C:/Users/HoaMi/Downloads/App human/static/images/jumpsuit.jpg', 
    'Quần salopette': 'C:/Users/HoaMi/Downloads/App human/static/images/salopette.jpg',
    'Áo khoác ngắn': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacngan.jpg' , 
    'Áo vest ngắn tay': 'C:/Users/HoaMi/Downloads/App human/static/images/aovestngantay.jpg',
    'Quần dài chuẩn': 'C:/Users/HoaMi/Downloads/App human/static/images/quandai.jpg',
    'Quần tây': 'C:/Users/HoaMi/Downloads/App human/static/images/quantay.png',
    'Quần yoga dài': 'C:/Users/HoaMi/Downloads/App human/static/images/quanyoga.jpg',
    'Quần jeans': 'C:/Users/HoaMi/Downloads/App human/static/images/jeans.jpg',
    'Quần jeans dài extra': 'C:/Users/HoaMi/Downloads/App human/static/images/jeansdai.jpg' , 
    'Quần ống suông': 'C:/Users/HoaMi/Downloads/App human/static/images/suông.jpg',
    'Vest': 'C:/Users/HoaMi/Downloads/App human/static/images/vest.jpg',
    'Áo khoác dài': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacday.jpg',
    'Đầm peplum': 'C:/Users/HoaMi/Downloads/App human/static/images/dampeplum.jpg',
    'Áo khoác ngắn': 'C:/Users/HoaMi/Downloads/App human/static/images/aokhoacngan.jpg',
    'Áo blazer rộng': 'C:/Users/HoaMi/Downloads/App human/static/images/blazer.jpg',
    'Áo sơ mi ôm': 'C:/Users/HoaMi/Downloads/App human/static/images/aosomi.png',
    'Váy ngắn rộng': 'C:/Users/HoaMi/Downloads/App human/static/images/vaymini.jpg',
    'Quần dài ống suông': 'C:/Users/HoaMi/Downloads/App human/static/images/suông.jpg',
}

# Load mô hình, scaler và label encoder
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Setup GUI
root = tk.Tk()
root.title("Fashion Recommender")

# Entries for user input
entry_labels = ["Shoulder Width", "Chest Width", "Waist", "Hips", "Total Height"]
entries = {}
for label in entry_labels:
    ttk.Label(root, text=label + ":").pack()
    entry = ttk.Entry(root)
    entry.pack(fill='x', expand=True)
    entries[label.replace(" ", "").lower()] = entry

# Gender selection
gender_var = tk.IntVar(value=0)  # Default to 'Female'
gender_frame = ttk.Frame(root)
ttk.Label(gender_frame, text="Gender:").pack(side=tk.LEFT)
ttk.Radiobutton(gender_frame, text="Female", variable=gender_var, value=0).pack(side=tk.LEFT)
ttk.Radiobutton(gender_frame, text="Male", variable=gender_var, value=1).pack(side=tk.LEFT)
gender_frame.pack()

# Prediction function
def predict():
    try:
        data = [float(entries[key].get()) for key in ['shoulderwidth', 'chestwidth', 'waist', 'hips', 'totalheight']]
        gender = gender_var.get()
        input_data = np.array([[gender] + data])
        features = scaler.transform(input_data)
        prediction = model.predict(features)
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        clothing_types = [label_encoder.classes_[i] for i in top_indices]
        result_var.set("Recommended Clothing: " + ", ".join(clothing_types))
        update_images(clothing_types)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to update image displays
def update_images(clothing_types):
    for i, ctype in enumerate(clothing_types):
        image_path = IMAGE_MAP.get(ctype, 'path_to_default_image.jpg')
        img = Image.open(image_path).resize((150, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img, master=root)
        image_labels[i].config(image=photo)
        image_labels[i].image = photo  # Keep a reference

# Setup result display and image labels
result_var = tk.StringVar()
result_label = ttk.Label(root, textvariable=result_var)
result_label.pack()
image_labels = [ttk.Label(root) for _ in range(3)]  # Image labels for the top 3 recommendations
for label in image_labels:
    label.pack()

# Buttons for predict and reset
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack()
reset_button = ttk.Button(root, text="Reset", command=lambda: [entry.delete(0, tk.END) for entry in entries.values()])
reset_button.pack()

# Main loop
root.mainloop()