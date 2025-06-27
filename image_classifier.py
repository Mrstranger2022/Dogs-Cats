import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# train_dir = 'D:\PORTFOLIO\PROJECTS\ImageClassifierusingKeras\datasets'

# To ensure that the path works even after providing relative path the following two lines of code are used
base_dir = os.path.dirname(__file__)  # This gets current script's folder
train_dir = os.path.join(base_dir, 'datasets')

datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)
train_data=datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='training'

)

val_data=datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation'

)

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit(train_data,validation_data=val_data,epochs=5)

plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='val')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()