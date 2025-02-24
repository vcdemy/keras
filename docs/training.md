# 4️⃣ 模型訓練與評估

## 🎯 訓練神經網路的基本概念

在建立 Keras 模型後，我們需要透過 **訓練（Training）** 來調整權重，使模型能夠對應輸入數據進行正確的預測。

✅ **訓練流程**：

1. **定義損失函數（Loss Function）**
2. **選擇優化器（Optimizer）**
3. **設定評估指標（Metrics）**
4. **執行訓練（Fit Model）**
5. **模型評估（Evaluate Model）**

---

## **✅ 編譯模型（Compile Model）**

Keras 需要在訓練前 **編譯（compile）** 模型，設定損失函數與優化器。

```python
from tensorflow import keras
from tensorflow.keras import layers

# 建立模型
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
```

✅ **`adam` 是一種常用的優化器，適合大多數深度學習應用。**

---

## **✅ 訓練模型（Model Training）**

我們可以使用 `fit()` 方法來訓練模型。

```python
import numpy as np

# 建立假設數據
x_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

# 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

✅ **`epochs=10` 表示訓練 10 次完整數據集，`batch_size=32` 表示每次更新 32 個樣本。**

---

## **✅ 模型評估（Evaluate Model）**

訓練完成後，我們可以使用測試數據來評估模型的表現。

```python
x_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, size=(20,))

# 評估模型
loss, acc = model.evaluate(x_test, y_test)
print(f"測試準確率：{acc:.2f}")
```

✅ **這將返回模型的損失與準確率，用來衡量模型的效能。**

---

## **✅ 使用 `validation_data` 進行驗證**

我們可以在訓練過程中使用 `validation_data` 來觀察模型的泛化能力。

```python
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

✅ **這樣可以看到訓練數據與驗證數據的準確率，避免過擬合。**

---

## **✅ 使用 Early Stopping 防止過擬合**

當訓練過程中發現 **驗證集準確率不再提升**，可以透過 `EarlyStopping` 來停止訓練。

```python
from tensorflow.keras.callbacks import EarlyStopping

# 設定 EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 訓練模型
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])
```
✅ **這樣可以防止模型過擬合，當 `val_loss` 連續 3 個 epochs 沒有改善時停止訓練。**

---

## 📝 **總結**

| **功能** | **語法** |
|----------|--------|
| **編譯模型** | `model.compile(optimizer, loss, metrics)` |
| **訓練模型** | `model.fit(x_train, y_train, epochs, batch_size)` |
| **評估模型** | `model.evaluate(x_test, y_test)` |
| **使用驗證數據** | `model.fit(x_train, y_train, validation_data=(x_test, y_test))` |
| **Early Stopping** | `EarlyStopping(monitor='val_loss', patience=3)` |

🚀 **現在你已經學會如何訓練與評估 Keras 模型！接下來，我們將學習 Keras 如何處理數據！** 😊

