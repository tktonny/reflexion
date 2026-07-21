# 开源唤醒词（openWakeWord，端侧离线）

用开源 **openWakeWord**（ONNX，完全离线、低耗电）做原生唤醒词。引擎 + hook 已实现并接进生产 `/conversation`（空闲时监听 → 命中 → 打开**日常助手** companion）。

## 现状（已完成）
- 引擎 `src/native/wakeWord.ts`：openWakeWord 三段式管线（melspectrogram → embedding → wakeword），复用现有 `expo-pcm-audio` 采集的 16kHz PCM。规格：mel 32 维、embedding 窗口 76 帧/步长 8 → 96 维、wakeword 窗口 16 个 embedding、阈值 0.5、连续 3 帧命中才触发（防抖）。
- Hook `src/hooks/useWakeWord.ts` + 接入 `app/conversation.tsx`（原生空闲态）。
- `onnxruntime-react-native` + `expo-asset` 已装，config plugin 已进 `app.json`。
- **优雅降级**：只要 onnxruntime 未链接、或模型不在设备上，`useWakeWord` 直接 no-op → 仍走"点屏幕开始"，**不影响 APK**。也因此在**模拟器上无法验证**（模拟器麦克风 HAL 不出声）——必须真机。

## 激活（在真机上补两步）

### 第 1 步：链接 onnxruntime 原生模块
config plugin 已加入 `app.json`。因为 `android/` 是预生成且有手改（Gradle 8.14.3 pin、`gradle.properties` 调优），推荐**增量** prebuild 后重编：
```bash
npx expo prebuild        # 应用 onnxruntime 的 config plugin（把原生包注册进 MainApplication）
# 若用 --clean 会重生成 android/，需重新施加 Gradle 8.14.3 pin 等手改
cd android && ./gradlew assembleRelease -x lint -x lintVitalRelease -PreactNativeArchitectures=arm64-v8a
```
> 注：onnxruntime-react-native 不带 `react-native.config.js`，autolink 会跳过——插件负责把它注册进原生工程。

### 第 2 步：放 3 个 ONNX 模型到设备
从 openWakeWord 拿模型（pip 包 `openwakeword/resources/models/` 或 GitHub Releases）：
- `melspectrogram.onnx`（共享）
- `embedding_model.onnx`（共享）
- 一个唤醒词模型，例如 `hey_jarvis_v0.1.onnx` → **重命名为 `wakeword.onnx`**（先用预训练词验证；自定义"Hey Aria"见下）

放到 App 私有目录 `<documentDirectory>/wakeword/`（即 `/data/data/com.reflexion.mirror/files/wakeword/`）。两种方式：
- **调试构建 adb**：`adb push *.onnx /data/local/tmp/ && adb shell run-as com.reflexion.mirror mkdir -p files/wakeword && adb shell run-as com.reflexion.mirror cp /data/local/tmp/*.onnx files/wakeword/`
- **或**加一个一次性**应用内下载**（你托管三个模型的 URL，首启用 `expo-file-system` 的 `File.downloadFileAsync` 拉到 `documentDirectory/wakeword/`）——要的话我给你写这个下载器（最省事）。

### 第 3 步：真机测试 + 调参
真机上说唤醒词 → 命中即打开助手。灵敏度在 `src/native/wakeWord.ts`：`THRESHOLD`（0.5）、`TRIGGER_HITS`（3）。太灵敏就升高，漏触发就降低。

## 自定义 "Hey Aria"
用 openWakeWord 的自动训练流程（合成 TTS 数据 + 负样本，官方 Colab <1 小时）导出 `.onnx`，替换 `wakeword.onnx` 即可。共享的 melspectrogram/embedding 不变。

## 备选
若不想上 ONNX/训练：**Vosk**（离线，语法限定关键词，无需训练，但更重）；或**保持点按开始**（对固定摆放的养老镜最省事最稳）。
