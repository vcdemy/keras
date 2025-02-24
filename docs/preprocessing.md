# 5️⃣ Keras 中的數據處理

## 🎯 為什麼數據處理很重要？

在訓練深度學習模型之前，我們通常需要對數據進行 **預處理（Preprocessing）**，以確保模型可以有效學習。

✅ **數據處理的關鍵步驟**：

1. **數據標準化（Normalization）**
2. **數據增強（Data Augmentation）**
3. **數據分割（Train / Validation / Test Split）**
4. **處理影像數據（Image Processing）**
5. **處理文本數據（Text Tokenization）**

---

## **✅ 數據標準化（Normalization）**

數據標準化可以讓所有特徵的數值範圍一致，提高模型訓練的穩定性。

```python
import numpy as np

# 生成隨機數據（模擬特徵）
data = np.random.randint(0, 255, (100, 10), dtype=np.float32)

# 標準化到 [0, 1] 區間
data_normalized = data / 255.0
```

✅ **常見的標準化方式包括 Min-Max Scaling（歸一化到 0-1）與 Z-score 標準化（均值 0，標準差 1）。**

---

## **✅ 數據增強（Data Augmentation）**

在影像處理中，我們可以通過 **旋轉、翻轉、縮放** 來增加數據多樣性。

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)
```

✅ **這有助於提升模型的泛化能力，防止過擬合（Overfitting）。**

---

## **✅ 訓練數據與測試數據分割**

一般來說，數據應該按照 **80% 訓練（Training）、10% 驗證（Validation）、10% 測試（Testing）** 來分割。

```python
from sklearn.model_selection import train_test_split

# 生成隨機數據
data = np.random.rand(1000, 10)
labels = np.random.randint(0, 2, size=(1000,))

# 分割數據
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

✅ **確保測試數據不被用於訓練，以檢驗模型的泛化能力。**

---

## **✅ 影像數據處理（Image Processing）**

在 Keras 中，可以使用 `ImageDataGenerator` 來讀取與增強影像數據。

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
```

✅ **這將自動讀取影像資料夾，並將影像縮放到 [0,1] 區間。**

---

## **✅ 處理文本數據（Text Tokenization）**

在 NLP 領域，我們需要將文本轉換為數字格式才能輸入到神經網路。

```python
from tensorflow.keras.preprocessing.text import Tokenizer

# 文字數據
documents = ["I love deep learning", "Keras makes it easy"]

# 建立 Tokenizer
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(documents)

# 轉換為數字序列
sequences = tokenizer.texts_to_sequences(documents)
print(sequences)
```

✅ **這可以將句子轉換為數字序列，以便神經網路處理。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **數據標準化** | `data / 255.0` |
| **影像增強** | `ImageDataGenerator(rotation_range=30, horizontal_flip=True)` |
| **數據分割** | `train_test_split(data, labels, test_size=0.2)` |
| **影像數據處理** | `flow_from_directory('dataset/train', target_size=(128,128))` |
| **文本 Tokenization** | `texts_to_sequences(documents)` |

🚀 **現在你已經學會如何在 Keras 中進行數據預處理！接下來，我們將學習 Keras 在 CNN（卷積神經網路）中的應用！** 😊

