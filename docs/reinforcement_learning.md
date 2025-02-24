# 9️⃣ 強化學習

## 🎯 什麼是強化學習？

強化學習（Reinforcement Learning, RL）是一種 **基於獎勵與懲罰的學習方法**，常用於 **遊戲 AI、機器人控制、自動駕駛** 等領域。

✅ **強化學習的核心概念**：

1. **代理（Agent）**：學習者，如機器人或 AI 模型。
2. **環境（Environment）**：代理與之互動的世界。
3. **動作（Action）**：代理可以採取的行為。
4. **狀態（State）**：環境在某個時間點的情況。
5. **獎勵（Reward）**：代理根據動作獲得的回饋。

---

## **✅ 建立 Q-learning 模型**

Q-learning 是一種基本的強化學習算法，我們可以使用 Keras 來建立 Q-learning 神經網路。

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 建立 Q-learning 模型
model = keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(4,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(2, activation='linear')
])

# 編譯模型
model.compile(optimizer='adam', loss='mse')
```

✅ **這是一個用於 OpenAI Gym CartPole 的基本 Q-learning 模型。**

---

## **✅ 訓練 Q-learning 模型**

在強化學習中，我們需要透過反覆試錯來學習最優策略。

```python
import gym

# 初始化環境
env = gym.make("CartPole-v1")

# 設定參數
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    done = False
    while not done:
        action = np.argmax(model.predict(state)[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
```

✅ **這段程式碼會讓 AI 嘗試學習如何在 CartPole 環境中取得較高的分數。**

---

## 📝 **總結**

| **概念** | **說明** |
|----------|--------|
| **代理（Agent）** | 學習與決策的 AI 模型 |
| **環境（Environment）** | AI 與之互動的世界，如遊戲、物理場景 |
| **動作（Action）** | 代理可以執行的行為 |
| **獎勵（Reward）** | 代理根據動作獲得的回饋 |

🚀 **現在你已經學會如何使用 Keras 來實作強化學習！接下來，我們將學習 Keras 在進階應用與專案示範中的應用！** 😊

