# 开源唤醒词（openWakeWord，端侧离线，已打进 APK）

用开源 **openWakeWord**（ONNX，完全离线、低耗电）做原生唤醒词。引擎 + hook + 接入 + **模型都已内置**。

## 现状（已完成，开箱即用）
- 引擎 `src/native/wakeWord.ts`：openWakeWord 三段式管线（melspectrogram → embedding → wakeword）。规格：mel 32 维；embedding 窗口 76 帧 / 步长 8 → 96 维；wakeword 窗口 16 个 embedding；阈值 0.5；连续 3 帧命中才触发（防抖）。复用现有 `expo-pcm-audio` 采集的 16kHz PCM。
- **模型已打进 APK**：`assets/wakeword/{melspectrogram,embedding_model,wakeword}.onnx`（共 ~3.7MB，来自 openWakeWord v0.5.1 Release；`wakeword.onnx` = 预训练 "hey jarvis"）。`metro.config.js` 把 `.onnx` 注册为可打包资源，引擎用 `expo-asset` 从内置资源加载 → **无需 adb / 设备目录 / 下载**。
- `onnxruntime-react-native` 原生库已随构建链接（arm64 APK 因此 ~52MB → ~80MB，加模型后 ~84MB）。
- Hook `src/hooks/useWakeWord.ts` + 接进 `app/conversation.tsx`（原生空闲 → 命中 → 打开日常助手 companion）。
- **优雅降级**：若某构建里 onnxruntime 未链接或加载失败，`createWakeWordEngine` 返回 null → 仍走"点屏幕开始"，不影响 App。

## 真机验证
- **必须真机**（模拟器麦克风 HAL 不出声，测不了检测）。装 APK → 生产 `/conversation` 空闲时说 **"hey jarvis"** → 应打开日常助手。
- 灵敏度在 `src/native/wakeWord.ts`：`THRESHOLD`（0.5）、`TRIGGER_HITS`（3）。太灵敏就升高，漏触发就降低。

## 换成自定义 "Hey Aria"
1. 用 openWakeWord 的自动训练流程（合成 TTS 正样本 + 负样本，官方 Colab <1 小时）训练并导出 `.onnx`。
2. 把导出的模型**替换** `assets/wakeword/wakeword.onnx`（共享的 melspectrogram/embedding 不动）。
3. 重新打包 v3 APK。

## 备选
若不想为唤醒词多背 ~28MB 的 onnxruntime：可把它做成**可选 flavor**（不带唤醒词的构建回到 ~52MB + 点按开始）；或用 **Vosk**（离线、语法关键词、无需训练但更重）；或**保持点按开始**（固定摆放的养老镜最省事最稳）。
