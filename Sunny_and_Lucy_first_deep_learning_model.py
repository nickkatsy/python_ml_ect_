import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt



class_names = ['Sunny','Lucy','both']
class_names_labels = {class_name: i for i, class_name in enumerate(class_names)}

image_size = (150,150)


def display_images(folder_path,label):
    plt.figure(figsize=(15,3))
    for i, filename in enumerate(os.listdir(folder_path)[:5]):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.title(f'{label}')
        plt.axis('off')
    plt.show()

# Pictures of the dogs
display_images('C:/ml/python/data/lucy_sunny/Sunny','Sunny')
display_images('C:/ml/python/data/lucy_sunny/Lucy','Lucy')
display_images('C:/ml/python/data/lucy_sunny/Sunny_and_Lucy','Both')


def load_images(folder_path,label):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load images for each class
sunny_images, sunny_labels = load_images('C:/ml/python/data/lucy_sunny/Sunny',class_names_labels['Sunny'])
lucy_images, lucy_labels = load_images('C:/ml/python/data/lucy_sunny/Lucy',class_names_labels['Lucy'])
both_images, both_labels = load_images('C:/ml/python/data/lucy_sunny/Sunny_and_Lucy',class_names_labels['both'])




# images for each Sunny,Lucy, and both
sunny_images, sunny_labels = load_images('C:/ml/python/data/lucy_sunny/Sunny', class_names_labels['Sunny'])
lucy_images, lucy_labels = load_images('C:/ml/python/data/lucy_sunny/Lucy', class_names_labels['Lucy'])
both_images, both_labels = load_images('C:/ml/python/data/lucy_sunny/Sunny_and_Lucy', class_names_labels['both'])

# Concatenate images and labels
X = np.concatenate([sunny_images,lucy_images,both_images],axis=0)
y = np.concatenate([sunny_labels,lucy_labels,both_labels],axis=0)

# train/test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


X_train = X_train/255.0
X_test = X_test/255.0

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,epochs=20,validation_split=0.20)

# loss/accuracy
test_loss, test_acc = model.evaluate(X_test,y_test)
print(f'Test Accuracy: {test_acc}')
print(f'test loss: {test_loss}')


# training history
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# predictions on the test set
predictions = model.predict(X_test)
y_pred = np.argmax(predictions,axis=1)

# classification report
print(classification_report(y_test,y_pred,target_names=class_names))


rand_indices = np.random.choice(len(X_test),size=min(15,len(X_test)),replace=False)

plt.figure(figsize=(15,3*len(rand_indices)))

for i, index in enumerate(rand_indices):
    plt.subplot(len(rand_indices),1,i+1)
    plt.imshow(X_test[index])
    predicted_class = class_names[int(y_pred[index])]
    true_class = class_names[int(y_test[index])]
    color = 'green' if predicted_class == true_class else 'red'
    plt.title(f'True: {true_class}|Predicted: {predicted_class}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()