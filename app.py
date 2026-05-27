import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Cấu hình trang
st.set_page_config(page_title="App Nhận Diện Các Loại Hoa", layout="wide", page_icon="🌻")

# Danh sách 8 loại hoa theo yêu cầu
FLOWER_CLASSES = [
    "Hoa Anh Đào", "Hoa Cúc", "Hoa Cẩm Tú Cầu", "Hoa Hồng", 
    "Hoa Hướng Dương", "Hoa Ly", "Hoa Sen", "Hoa Tulip"
]

# Tự động tạo thư mục rỗng nếu chưa có để người dùng bỏ ảnh vào
DATA_DIR = "data_hoa"
for flower in FLOWER_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, flower), exist_ok=True)

@st.cache_resource
def load_local_model():
    """Chỉ load model từ bộ nhớ cục bộ, bỏ qua Github URL rườm rà"""
    model_path = "flower_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_local_model()

st.sidebar.title("Thanh Công Cụ 🛠️")
app_mode = st.sidebar.selectbox("Chọn chức năng", ["Dự đoán qua Webcam/Ảnh", "Huấn luyện mô hình (Training)"])

if app_mode == "Dự đoán qua Webcam/Ảnh":
    st.title("🌻 Nhận diện Các Loại Hoa (CNN)")
    
    if model is None:
        st.warning("⚠️ Chưa tìm thấy mô hình `flower_model.h5`. Vui lòng chuyển sang tab 'Huấn luyện mô hình' để training trước!")
    else:
        st.success("✅ Đã tải mô hình nhận diện hoa thành công!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Sử dụng Camera")
            camera_image = st.camera_input("Chụp ảnh từ Webcam")
            
        with col2:
            st.markdown("### 2. Tải ảnh từ máy")
            uploaded_file = st.file_uploader("Chọn file ảnh (jpg, png)", type=["jpg", "jpeg", "png"])
            
        img_to_predict = camera_image if camera_image else uploaded_file
        
        if img_to_predict is not None:
            image = Image.open(img_to_predict).convert('RGB')
            st.image(image, caption="Ảnh đầu vào", use_column_width=True)
            
            # Tiền xử lý ảnh
            img_width, img_height = 200, 200
            img_resized = image.resize((img_width, img_height))
            img_array = np.array(img_resized)
            
            img_for_prediction = img_array.astype('float32') / 255.0
            img_for_prediction = np.expand_dims(img_for_prediction, axis=0)
            
            if st.button("🔍 Tiến hành Dự đoán", type="primary"):
                with st.spinner("Đang phân tích..."):
                    preds = model.predict(img_for_prediction)
                    digit = int(np.argmax(preds))
                    confidence = np.max(preds) * 100
                    
                    predicted_name = FLOWER_CLASSES[digit]
                    
                    st.success(f"**Kết quả dự đoán: {predicted_name}**")
                    st.info(f"Độ tin cậy (Confidence): {confidence:.2f}%")
            

elif app_mode == "Huấn luyện mô hình (Training)":
    st.title("⚙️ Huấn luyện Mô hình Nhận diện Hoa")
    
    st.info(f"**Hướng dẫn:** Ứng dụng đã tự động tạo thư mục `{DATA_DIR}`. Hãy vào thư mục này, bạn sẽ thấy 8 thư mục con của 8 loại hoa. Hãy copy ít nhất 15-20 ảnh gốc của mỗi loại vào đúng thư mục tương ứng. Code sẽ tự động biến đổi (xoay, lật, zoom) để tạo ra hàng trăm ảnh trong quá trình train.")
    
    epochs = st.number_input("Số lần học (Epochs):", min_value=1, max_value=100, value=15)
    batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=16)
    
    if st.button("🚀 Bắt đầu Huấn luyện"):
        # Kiểm tra xem có ảnh trong thư mục chưa
        total_images = sum([len(files) for r, d, files in os.walk(DATA_DIR)])
        if total_images < 10:
            st.error(f"Thư mục `{DATA_DIR}` đang trống hoặc có quá ít ảnh. Vui lòng thêm ảnh vào các thư mục con trước khi huấn luyện!")
        else:
            with st.spinner("Đang tiến hành đọc dữ liệu và huấn luyện... quá trình này có thể mất vài phút."):
                img_width, img_height = 200, 200
                
                # Data Augmentation: Thay đổi góc nhìn, lật, xoay để tăng cường dữ liệu
                train_datagen = ImageDataGenerator(
                    rescale=1.0/255, 
                    rotation_range=40,          # Tăng góc xoay lên 40 độ
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.3,             # Zoom cận cảnh hoa
                    horizontal_flip=True,       # Lật ngang
                    fill_mode="nearest",
                    validation_split=0.2        # Trích 20% làm tập kiểm thử
                )                                                                                                                  
                
                train_generator = train_datagen.flow_from_directory(
                    DATA_DIR,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    classes=FLOWER_CLASSES,     # Ép cứng thứ tự class
                    subset='training'
                )
                
                val_generator = train_datagen.flow_from_directory(
                    DATA_DIR,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    classes=FLOWER_CLASSES,
                    subset='validation'
                )

                # Xây dựng cấu trúc mô hình CNN cho 8 classes
                model = Sequential([
                    Conv2D(32, (3,3), activation="relu", input_shape=(img_width, img_height, 3)), 
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(len(FLOWER_CLASSES), activation="softmax") # Output 8 loại hoa
                ])                                                                                                         
                
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                
                st.text("Cấu trúc mô hình:")
                model.summary(print_fn=lambda x: st.text(x))

                # Bắt đầu Train
                history = model.fit(
                    train_generator, 
                    validation_data=val_generator,
                    epochs=epochs
                )
                
                # Lưu model đè lên file cũ
                model.save("flower_model.h5")
                st.success("✅ Huấn luyện hoàn tất! Mô hình đã được lưu thành `flower_model.h5`.")

                # Vẽ biểu đồ kết quả
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label="Độ chính xác huấn luyện")
                ax.plot(history.history['val_accuracy'], label="Độ chính xác kiểm thử (Validation)")
                ax.set_xlabel("Số lần học (Epochs)")
                ax.set_ylabel("Độ chính xác")
                ax.legend()
                st.pyplot(fig)
