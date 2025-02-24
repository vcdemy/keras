# 🔟 進階應用與專案示範

## 🎯 Keras 的進階應用

現在我們已經學會了 Keras 的基礎與核心概念，接下來，我們將探索 Keras 在實際專案中的應用。

✅ **進階應用領域**：

1. **影像分類（Image Classification）**
2. **物件偵測（Object Detection）**
3. **自然語言處理（NLP）**
4. **時間序列預測（Time Series Forecasting）**
5. **生成對抗網路（GANs）**

---

## **✅ 影像分類專案示範（Image Classification）**

我們可以使用 Keras 來建立一個影像分類模型，例如 CIFAR-10 數據集。

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# 載入 CIFAR-10 數據集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 正規化數據
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立 CNN 模型
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

✅ **這是一個適用於影像分類的 CNN 模型，適合應用於物件辨識等任務。**

---

## **✅ NLP 應用（情感分析）**

Keras 也可以用於 **自然語言處理（NLP）**，例如情感分析。

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 建立訓練數據
documents = ["I love this movie!", "This film is terrible.", "Amazing story and characters."]
labels = [1, 0, 1]  # 1: 正面情緒, 0: 負面情緒

# Tokenization
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)

# 填充序列
padded_sequences = pad_sequences(sequences, maxlen=10)
```

✅ **這樣可以將文字轉換為數值表示，以便輸入到 NLP 模型中進行訓練。**

---

## 📝 **總結**

| **應用領域** | **範例** |
|----------|--------|
| **影像分類** | `CNN` 處理 CIFAR-10 數據集 |
| **自然語言處理（NLP）** | 文字 Tokenization 與情感分析 |
| **時間序列預測** | `LSTM` 處理金融數據預測 |
| **物件偵測** | `YOLO` 進行即時影像分析 |
| **生成對抗網路（GANs）** | 影像生成與風格轉換 |

🚀 **恭喜你！你已經完成了 Keras 教學課程！現在你可以使用 Keras 構建自己的深度學習專案了！🎉** 😊

