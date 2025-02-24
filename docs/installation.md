# 2️⃣ 安裝與基本使用

## 🎯 如何安裝 Keras？

Keras 是 **基於 TensorFlow** 的深度學習框架，因此我們需要安裝 TensorFlow 來使用 Keras。

✅ **安裝 TensorFlow 和 Keras**

```bash
pip install tensorflow keras
```

✅ **檢查安裝是否成功**

```python
import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)
```

🚀 **如果成功顯示 TensorFlow 和 Keras 版本，表示安裝完成！**

---

## 🎯 建立第一個 Keras 模型

我們使用 **Sequential API** 來建立一個簡單的神經網路。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 建立模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 查看模型結構
model.summary()
```

✅ **這是一個簡單的全連接神經網路（DNN），適用於二元分類任務。**

---

## 🎯 訓練與評估模型

我們使用 **假設的訓練數據** 來訓練模型。

```python
import numpy as np

# 建立假設數據
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

# 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

✅ **這將訓練模型 10 個 epochs，每次 batch 為 32。**

### **評估模型**

```python
x_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, size=(20,))

loss, acc = model.evaluate(x_test, y_test)
print(f"測試準確率：{acc:.2f}")
```

✅ **這將在測試數據上評估模型的表現。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **安裝 Keras** | `pip install tensorflow keras` |
| **建立模型** | `keras.Sequential([...])` |
| **編譯模型** | `model.compile(optimizer, loss, metrics)` |
| **訓練模型** | `model.fit(x_train, y_train, epochs, batch_size)` |
| **評估模型** | `model.evaluate(x_test, y_test)` |

🚀 **現在你已經學會如何安裝 Keras 並建立第一個深度學習模型！接下來，我們將深入探討 Keras 的模型結構與 API！** 😊

