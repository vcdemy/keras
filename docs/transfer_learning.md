# 8️⃣ 轉移學習

## 🎯 什麼是轉移學習？

轉移學習（Transfer Learning）是一種 **利用預訓練模型** 來加速新模型訓練的技術，適用於 **影像分類、物件偵測、NLP 等應用**。

✅ **轉移學習的優勢**：

1. **減少訓練時間**：直接使用已訓練好的權重。
2. **適用於小型數據集**：不需要大量標註數據。
3. **提高模型準確率**：使用大型數據集訓練的特徵。

---

## **✅ 使用預訓練模型（VGG16）**

我們可以使用 Keras 內建的 VGG16 模型進行影像分類。

```python
from tensorflow import keras
from tensorflow.keras.applications import VGG16

# 載入 VGG16 預訓練模型（去除頂部全連接層）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 凍結預訓練層
for layer in base_model.layers:
    layer.trainable = False
```

✅ **這樣可以保留 VGG16 學到的特徵，避免過度訓練。**

---

## **✅ 加入自訂分類層**

我們可以在 VGG16 頂部加入自訂的全連接層。

```python
from tensorflow.keras import layers, models

# 建立新模型
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

✅ **這樣可以讓模型適應新的數據集，進行分類任務。**

---

## **✅ 訓練轉移學習模型**

我們可以使用 ImageDataGenerator 來處理影像數據並訓練模型。

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 設定數據增強
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

# 訓練模型
model.fit(train_generator, epochs=10)
```

✅ **這樣可以讓模型學習新數據的特徵，提高分類準確率。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **載入 VGG16 預訓練模型** | `VGG16(weights='imagenet', include_top=False)` |
| **凍結預訓練層** | `layer.trainable = False` |
| **添加自訂分類層** | `model.add(Dense(256, activation='relu'))` |
| **使用數據增強** | `ImageDataGenerator(rotation_range=30, horizontal_flip=True)` |

🚀 **現在你已經學會如何使用 Keras 進行轉移學習！接下來，我們將學習 Keras 在強化學習（Reinforcement Learning）中的應用！** 😊

