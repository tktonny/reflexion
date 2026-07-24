# 依赖：dashscope >= 1.23.9，pyaudio
import os
import base64
import time
import threading
import pyaudio
from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeCallback, OmniRealtimeConversation
import dashscope

# 配置参数：地址、API Key、音色、模型、模型角色
# 指定地域，设为 cn 表示中国内地（北京），设为 intl 表示国际（新加坡）
region = os.getenv('DASHSCOPE_REGION', 'cn')
base_domain = 'dashscope.aliyuncs.com' if region == 'cn' else 'dashscope-intl.aliyuncs.com'
url = f'wss://{base_domain}/api-ws/v1/realtime'

# 优先从环境变量读取 API Key，避免把密钥硬编码在代码里
# export DASHSCOPE_API_KEY="sk-xxx"
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
if not dashscope.api_key:
    raise RuntimeError('未检测到 DASHSCOPE_API_KEY，请先在环境变量中配置 API Key。')

# 指定音色
voice = 'Cherry'
# 指定模型
model = 'qwen3.5-omni-plus-realtime'
# 指定模型角色
instructions = "你是个人助理小云，请用幽默风趣的方式回答用户的问题"

class SimpleCallback(OmniRealtimeCallback):
    def __init__(self, pya):
        self.pya = pya
        self.out = None
        self.is_open = False
        self.last_error = None
        self.close_info = None
        self.open_event = threading.Event()

    def on_open(self):
        print('[WS] connected')
        self.is_open = True
        self.open_event.set()
        # 初始化音频输出流
        self.out = self.pya.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )

    def on_close(self, code=None, reason=None):
        self.is_open = False
        self.close_info = {'code': code, 'reason': reason}
        print(f'[WS] closed code={code} reason={reason}')

    def on_error(self, error):
        self.last_error = str(error)
        print(f'[WS] error: {error}')

    def on_event(self, response):
        event_type = response.get('type')
        if event_type == 'response.audio.delta':
            # 播放音频
            if self.out is not None:
                self.out.write(base64.b64decode(response['delta']))
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            # 打印转录文本
            print(f"[User] {response.get('transcript', '')}")
        elif event_type == 'response.audio_transcript.done':
            # 打印助手回复文本
            print(f"[LLM] {response.get('transcript', '')}")
        elif event_type == 'error':
            print(f'[Server Error] {response}')


# 1. 初始化音频设备
pya = pyaudio.PyAudio()
mic = None
conv = None

try:
    # 2. 创建回调函数和会话
    callback = SimpleCallback(pya)
    conv = OmniRealtimeConversation(model=model, callback=callback, url=url)

    # 3. 建立连接
    print(f'[Config] region={region} url={url} model={model}')
    conv.connect()

    # 等待 WebSocket 真正打开，避免连接还没 ready 就开始发音频
    if not callback.open_event.wait(timeout=10):
        raise RuntimeError(
            f'WebSocket 连接未在 10 秒内建立。last_error={callback.last_error}, close_info={callback.close_info}'
        )

    # 4. 配置会话
    conv.update_session(
        output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
        voice=voice,
        instructions=instructions,
    )

    # 给服务端一点时间完成 session 初始化
    time.sleep(0.5)

    # 5. 初始化音频输入流
    mic = pya.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)

    # 6. 主循环处理音频输入
    print('对话已开始，对着麦克风说话 (Ctrl+C 退出)...')
    while True:
        if not callback.is_open:
            raise RuntimeError(
                f'WebSocket 已关闭，停止发送音频。last_error={callback.last_error}, close_info={callback.close_info}'
            )

        audio_data = mic.read(3200, exception_on_overflow=False)
        try:
            conv.append_audio(base64.b64encode(audio_data).decode())
        except Exception as e:
            raise RuntimeError(
                f'发送音频失败: {e}; last_error={callback.last_error}, close_info={callback.close_info}'
            ) from e
        time.sleep(0.01)

except KeyboardInterrupt:
    print('\n对话结束')
except Exception as e:
    print(f'[Fatal] {e}')
finally:
    if conv is not None:
        try:
            conv.close()
        except Exception:
            pass
    if mic is not None:
        try:
            mic.close()
        except Exception:
            pass
    if 'callback' in locals() and callback.out is not None:
        try:
            callback.out.close()
        except Exception:
            pass
    pya.terminate()