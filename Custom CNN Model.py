import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, models


# 1. تحديد الإعدادات الأساسية
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 2 

# مسار الفولدر الأساسي المحتوي على الكلاسات (عدله حسب مساره الفعلي على جهازك)
DATA_DIR = './original_data' 

# 2. خطوة قراءة وتحضير البيانات (Data Pipeline)
# قراءة داتا التدريب (Training)
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, # تقسيم 20% للتحقق
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# قراءة داتا التحقق (Validation)
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# عمل Normalize لقيم البيكسل لتصبح بين 0 و 1 أثناء القراءة السريعة
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))


# 3. بناء هيكل الـ CNN Custom Model
def create_oral_disease_model():
    model = models.Sequential([
        # الطبقة التلافيفية الأولى + MaxPooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        layers.MaxPooling2D((2, 2)),
        
        # الطبقة التلافيفية الثانية
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # الطبقة التلافيفية الثالثة
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # تحويل البيانات إلى 1D
        layers.Flatten(),
        
        # طبقة Dense لتعلم العلاقات المعقدة
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # لتقليل الـ Overfitting
        
        # طبقة الإخراج النهائية
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# 4. إنشاء الموديل وعمل الـ Compile بالفاصلة الناقصة
model = create_oral_disease_model()

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9), # تم تصليح الفاصلة هنا
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# عرض هيكل الموديل
model.summary()

# 5. تدريب الموديل
model.fit(
    train_data, 
    epochs=10, 
    validation_data=val_data
)

# 6. حفظ الموديل لاستخدامه في الـ FastAPI لاحقاً
model.save('oral_disease_model.h5')
print("Model saved successfully as oral_disease_model.h5")