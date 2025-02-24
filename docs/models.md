# 3️⃣ Keras 基本模型

## 🎯 Keras 的模型結構

在 Keras 中，我們可以使用 **Sequential API** 或 **Functional API** 來構建深度學習模型。

✅ **Sequential API（順序模型）**

這種方式適合 **線性堆疊** 的模型，每一層按照順序堆疊。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 建立順序模型
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
```

✅ **適用於大部分的神經網路，如 MLP（多層感知機）。**

---

✅ **Functional API（函數式模型）**

Functional API 允許更靈活的模型設計，例如 **多輸入、多輸出、跳躍連接**。

```python
# 定義輸入層
inputs = keras.Input(shape=(10,))

# 隱藏層
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)

# 輸出層
outputs = layers.Dense(1, activation='sigmoid')(x)

# 建立模型
model = keras.Model(inputs, outputs)

model.summary()
```

✅ **適用於複雜的模型，如殘差網路（ResNet）、注意力機制等。**

---

## 🎯 增強模型功能：Dropout 與 Batch Normalization

✅ **Dropout（隨機失活）**

Dropout 會隨機關閉部分神經元，防止 **過擬合（Overfitting）**。

```python
model.add(layers.Dropout(0.5))
```

✅ **Batch Normalization（批次標準化）**

Batch Normalization 可加速訓練，穩定梯度下降。

```python
model.add(layers.BatchNormalization())
```

---

## 📝 **總結**

| **概念** | **用途** |
|----------|--------|
| **Sequential API** | 適用於簡單的線性堆疊模型 |
| **Functional API** | 適用於複雜的多輸入、多輸出模型 |
| **Dropout** | 防止過擬合，提高泛化能力 |
| **Batch Normalization** | 加速訓練，提高模型穩定性 |

🚀 **現在你已經學會如何使用 Keras 建立不同類型的模型！接下來，我們將學習如何訓練與評估模型！** 😊

