# 7️⃣ RNN（循環神經網路）

## 🎯 什麼是 RNN？

循環神經網路（Recurrent Neural Network, RNN）適用於 **序列數據（Sequential Data）**，例如 **時間序列、自然語言處理（NLP）、語音辨識** 等。

✅ **RNN 的核心概念**：

1. **時間步長（Timesteps）**：保留過去資訊以影響未來預測。
2. **隱藏狀態（Hidden State）**：存儲過去輸入的資訊。
3. **長短期記憶（LSTM）與門控循環單元（GRU）**：解決 RNN 訓練時的長期依賴問題。

---

## **✅ 建立一個 RNN 模型**

我們使用 Keras 來建立一個基本的 RNN 模型。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 建立 RNN 模型
model = keras.Sequential([
    layers.SimpleRNN(64, activation='relu', input_shape=(10, 1)),
    layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

✅ **這是一個適用於時間序列分類的基本 RNN 結構。**

---

## **✅ LSTM（長短期記憶網路）**

LSTM 是 RNN 的改進版本，解決了 **梯度消失（Vanishing Gradient）** 的問題。

```python
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(10, 1)),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])
```

✅ **LSTM 適用於較長的序列數據，例如自然語言處理（NLP）。**

---

## **✅ GRU（門控循環單元）**

GRU 是 LSTM 的簡化版本，運算較快，效果相近。

```python
model = keras.Sequential([
    layers.GRU(64, return_sequences=True, input_shape=(10, 1)),
    layers.GRU(32),
    layers.Dense(1, activation='sigmoid')
])
```

✅ **GRU 在小型數據集上通常比 LSTM 更高效。**

---

## **✅ 訓練 RNN 模型**

我們使用隨機生成的數據來訓練 RNN。

```python
import numpy as np

# 生成隨機數據
x_train = np.random.rand(1000, 10, 1)
y_train = np.random.randint(0, 2, size=(1000,))

# 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

✅ **這將訓練 10 個 epochs，每次 batch 為 32。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **RNN 層** | `SimpleRNN(units, activation)` |
| **LSTM 層** | `LSTM(units, return_sequences)` |
| **GRU 層** | `GRU(units, return_sequences)` |
| **模型訓練** | `model.fit(x_train, y_train, epochs, batch_size)` |

🚀 **現在你已經學會如何使用 Keras 建立 RNN！接下來，我們將學習轉移學習（Transfer Learning）！** 😊

