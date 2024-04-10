import os
import sys
import cv2 
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Egg crack checking
def load_data(data_directory, img_width, img_height, batch_size):
    train_data_directory = os.path.join(data_directory, 'train')
    validation_data_directory = os.path.join(data_directory, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_and_save_model(data_directory, img_width, img_height, batch_size, epochs, model_path):
    train_generator, validation_generator = load_data(data_directory, img_width, img_height, batch_size)
    input_shape = (img_width, img_height, 3)

    model = create_model(input_shape)
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size)

    model.save(model_path)


def predict_image(model_path, image_path, img_width, img_height):
    model = load_model(model_path)
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return 'GOOD' if prediction[0][0] > 0.5 else 'CRACKED'


#Egg yolk checking 
def calculate_air_sac_size(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        air_sac_size = h / img.shape[0]
        return air_sac_size
    except Exception as e:
        print(f"Error: {e}")


def age_approximation(s):
    if s == 'VERY FRESH':
        return "0-1 WEEKS"
    elif s == "FRESH":
        return "1-2 WEEKS"
    elif s == "MODERATELY FRESH":
        return "2-3 WEEKS"
    elif s == "NOT FRESH":
        return "3+ WEEKS"
    else: 
        return s

def egg_freshness(air_sac_size):
    if air_sac_size is not None:
        if air_sac_size < 0.25:
            return "VERY FRESH"
            # return "0-1 WEEKS"
        elif 0.25 <= air_sac_size < 0.5:
            return "FRESH"
            # return "1-2 WEEKS"
        elif 0.5 <= air_sac_size < 0.75:
            return "MODERATELY FRESH"
            # return "2-3 WEEKS"
        else:
            return "NOT FRESH"
            # return "3+ WEEKS"
    else:
        return "Unable to process image"

def save_uploaded_file(uploaded_file):
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_image.jpg"

def main():
    try:
        st.set_page_config(page_title="Egg Quality Analysis")
        st.title("EGG QUALITY CONTROL")
        st.write("Welcome to the Egg Freshness and Quality Analysis checker")
        st.divider()
        option = st.radio("Choose option", ("Home", "Dev Options", "Info"), index=0)

        if option == "Home":
            st.sidebar.subheader("Quality Analysis Options")
            check_egg = st.sidebar.checkbox("Check Egg")
            if check_egg:
                image_path = st.camera_input("Take a Picture")
                if st.button("Check Egg", key="check_egg_button"):
                    if image_path is not None:
                        st.image(image_path, caption="Image", use_column_width=True)
                        image_path = save_uploaded_file(image_path)
                    print(calculate_air_sac_size(image_path))
                    freshness = egg_freshness(calculate_air_sac_size(image_path))
                    #meow
                    # Change path to egg_classifier.h5 path:
                    result = predict_image('D:\Projects\MSRIT-EGG\EggFinalGui\Egg\egg_classifier.h5', image_path=image_path, img_height=150, img_width=150)
                    st.divider()
                    # TABLE OF RESULT
                    df = pd.DataFrame({
                        "Freshness": [freshness],
                        "Shell": [result],
                        "Approx Age": [age_approximation(freshness)]
                    })
                    st.dataframe(df , hide_index=True)
                    #PLAIN TEXT RESULT
                    # st.write("Freshness:", freshness)
                    # st.write("Shell:", result)
                    # st.write("Approx Age:", age_approximation(freshness))
                else:
                    st.write("")

        elif option == "Dev Options":
            st.sidebar.subheader("Developer Options")
            train_model = st.sidebar.checkbox("Train Model")
            if train_model:
                st.divider()
                st.write("Training Model... this may take a while")
                #meow
                # Change path to data and egg_classifier.h5 path
                train_and_save_model('D:\Projects\MSRIT-EGG\EggFinalGui\Egg\data', 150, 150, 32, 20, 'D:\Projects\MSRIT-EGG\EggFinalGui\Egg\egg_classifier.h5')
                st.divider()
                st.write("Model trained successfully!")

        elif option == "Info":   
            st.empty()
            st.divider()
            st.write("The Egg Freshness Analysis system checks for freshness and quality of poultry eggs")
            st.write("Using this, one may check for various factors affecting poultry eggs like: Freshness, Egg Shell Quality, Egg Age\n")
            st.write("This works on a system of analysis of Egg density, egg shell screening, and analysis of airsac within egg")
            st.write("It uses a simple user friendly interface which makes it easy for anyone to use, and holds potential for growth to the industrial level as well")
            st.write("It hosts a highly specialized image processing system, which eliminates user error of taking images, though this may lead to software trying to find any presence of an egg in an image in some rare cases")

    except:
        st.write("An error occurred, try again")

if __name__ == "__main__":
    main()
