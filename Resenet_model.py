import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


IMG_HIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 2

DATA_DIR = './original_data'

train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,   
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)


def preprocess_data(x,y):
    x= tf.keras.applications.resnet50.preprocess_input(x)
    return x,y


train_data = train_data.map(preprocess_data)
val_data = val_data.map(preprocess_data)

def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HIGHT, IMG_WIDTH, CHANNELS))
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

model = create_resnet_model()

model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_data, validation_data=val_data, epochs=10)

model.save('resnet_model.h5')
print("Model saved as resnet_model.h5")