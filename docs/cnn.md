# 6️⃣ CNN（卷積神經網路）

## 🎯 什麼是 CNN？

卷積神經網路（Convolutional Neural Network, CNN）是一種專門用於 **影像處理** 的神經網路，能夠學習影像的 **特徵（Features）**，如邊緣、紋理、形狀等。

✅ **CNN 的核心概念**：

1. **卷積層（Convolutional Layer）**
2. **池化層（Pooling Layer）**
3. **全連接層（Fully Connected Layer）**

---

## **✅ 建立一個 CNN 模型**

我們使用 Keras 來建立一個基本的 CNN。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 建立 CNN 模型
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

✅ **這是一個適用於 MNIST 數字分類的基本 CNN 結構。**

---

## **✅ 什麼是卷積層？（Convolutional Layer）**

卷積層透過 **濾波器（Filters）** 來擷取影像特徵。

```python
layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))
```

✅ **這表示使用 32 個 3×3 的濾波器來處理影像。**

---

## **✅ 什麼是池化層？（Pooling Layer）**

池化層可以 **降低影像尺寸**，減少計算成本，並保留重要特徵。

```python
layers.MaxPooling2D((2,2))
```

✅ **這表示每 2×2 區域取最大值，以減少影像尺寸。**

---

## **✅ 訓練 CNN 模型**

我們使用 MNIST 數據集來訓練這個 CNN。

```python
from tensorflow.keras.datasets import mnist
import numpy as np

# 載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 正規化數據
x_train, x_test = x_train / 255.0, x_test / 255.0

# 增加維度，讓 CNN 可處理
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
```

✅ **這會訓練 5 個 epochs，並使用 32 個 batch size 來更新權重。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **卷積層** | `Conv2D(filters, kernel_size, activation)` |
| **池化層** | `MaxPooling2D(pool_size)` |
| **模型訓練** | `model.fit(x_train, y_train, epochs, batch_size)` |

🚀 **現在你已經學會如何使用 Keras 建立 CNN！接下來，我們將學習 RNN（循環神經網路）的應用！** 😊

