import AVFoundation
import ExpoModulesCore

// Streaming PCM bridge for v3 (direct Qwen realtime WS). Mirrors the Android module:
//  - Capture: mic -> PCM16 mono @ 16 kHz -> base64 -> `onAudioChunk`.
//  - Playback: base64 PCM16 mono @ 24 kHz enqueued via play(), scheduled on an AVAudioPlayerNode.
//  - Half-duplex: setCaptureMuted(true) drops captured frames during assistant playback.
//  Uses AVAudioSession .playAndRecord / .voiceChat so the OS applies echo cancellation.
public class ExpoPcmAudioModule: Module {
  private let engine = AVAudioEngine()
  private let player = AVAudioPlayerNode()
  private var playerAttached = false // attach/connect the node exactly once — re-attaching crashes

  private var captureConverter: AVAudioConverter?
  private var captureFormat: AVAudioFormat?   // 16 kHz mono Int16 (output of the input converter)
  private var playbackFormat: AVAudioFormat?  // 24 kHz mono Float32 (player node format)

  private var isCapturing = false
  private var isPlaying = false
  private var captureMuted = false
  private let captureSampleRate: Double = 16000
  private let playbackSampleRate: Double = 24000

  // Unplayed playback backlog, so JS un-mutes the mic only after the assistant truly stops talking.
  private let framesLock = NSLock()
  private var pendingPlaybackFrames: Int64 = 0

  public func definition() -> ModuleDefinition {
    Name("ExpoPcmAudio")

    Events("onAudioChunk")

    AsyncFunction("start") { (sampleRate: Int, promise: Promise) in
      do {
        try self.startAudio()
        promise.resolve(nil)
      } catch {
        self.stopAudio()
        promise.reject("ERR_PCM_START", error.localizedDescription)
      }
    }

    AsyncFunction("stop") { (promise: Promise) in
      self.stopAudio()
      promise.resolve(nil)
    }

    Function("play") { (base64: String) in
      self.enqueue(base64: base64)
    }

    Function("clearPlayback") {
      self.player.stop()
      self.setPendingFrames(0)
      if self.isPlaying { self.player.play() }
    }

    Function("setCaptureMuted") { (muted: Bool) in
      self.captureMuted = muted
    }

    Function("getPlaybackBacklogMs") { () -> Double in
      self.framesLock.lock()
      let frames = self.pendingPlaybackFrames
      self.framesLock.unlock()
      return Double(frames) / self.playbackSampleRate * 1000.0
    }

    OnDestroy {
      self.stopAudio()
    }
  }

  private func startAudio() throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
    try session.setPreferredSampleRate(playbackSampleRate)
    try session.setActive(true)

    // ---- playback graph: player -> mainMixer at 24 kHz mono float (attach/connect ONCE) ----
    guard let pbFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: playbackSampleRate, channels: 1, interleaved: false) else {
      throw NSError(domain: "ExpoPcmAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad playback format"])
    }
    playbackFormat = pbFormat
    if !playerAttached {
      engine.attach(player)
      engine.connect(player, to: engine.mainMixerNode, format: pbFormat)
      playerAttached = true
    }

    // ---- capture: input tap -> converter -> 16 kHz mono Int16 ----
    let input = engine.inputNode
    let inputFormat = input.inputFormat(forBus: 0)
    guard let capFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: captureSampleRate, channels: 1, interleaved: true) else {
      throw NSError(domain: "ExpoPcmAudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "bad capture format"])
    }
    captureFormat = capFormat
    captureConverter = AVAudioConverter(from: inputFormat, to: capFormat)

    input.installTap(onBus: 0, bufferSize: 3200, format: inputFormat) { [weak self] buffer, _ in
      self?.handleInput(buffer: buffer)
    }

    engine.prepare()
    try engine.start()
    setPendingFrames(0)
    player.play()
    isCapturing = true
    isPlaying = true
  }

  private func handleInput(buffer: AVAudioPCMBuffer) {
    guard isCapturing, !captureMuted, let converter = captureConverter, let outFormat = captureFormat else { return }
    let ratio = outFormat.sampleRate / buffer.format.sampleRate
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 512)
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }

    var fed = false
    var error: NSError?
    let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
      if fed {
        outStatus.pointee = .noDataNow
        return nil
      }
      fed = true
      outStatus.pointee = .haveData
      return buffer
    }
    if status == .error || outBuffer.frameLength == 0 { return }

    guard let channel = outBuffer.int16ChannelData else { return }
    let byteCount = Int(outBuffer.frameLength) * MemoryLayout<Int16>.size
    let data = Data(bytes: channel[0], count: byteCount)
    sendEvent("onAudioChunk", ["data": data.base64EncodedString()])
  }

  private func enqueue(base64: String) {
    guard isPlaying, let data = Data(base64Encoded: base64), !data.isEmpty else { return }
    guard let srcFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: playbackSampleRate, channels: 1, interleaved: true),
          let pbFormat = playbackFormat else { return }

    let frames = AVAudioFrameCount(data.count / MemoryLayout<Int16>.size)
    if frames == 0 { return }
    guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: frames) else { return }
    srcBuffer.frameLength = frames
    data.withUnsafeBytes { raw in
      if let base = raw.baseAddress, let dst = srcBuffer.int16ChannelData {
        memcpy(dst[0], base, Int(frames) * MemoryLayout<Int16>.size)
      }
    }

    // Convert Int16 -> Float32 (player node format) and schedule.
    guard let floatBuffer = AVAudioPCMBuffer(pcmFormat: pbFormat, frameCapacity: frames) else { return }
    floatBuffer.frameLength = frames
    if let src = srcBuffer.int16ChannelData, let dst = floatBuffer.floatChannelData {
      let s = src[0]
      let d = dst[0]
      for i in 0..<Int(frames) {
        d[i] = Float(s[i]) / 32768.0
      }
    }

    addPendingFrames(Int64(frames))
    player.scheduleBuffer(floatBuffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
      self?.addPendingFrames(-Int64(frames))
    }
    if !player.isPlaying { player.play() }
  }

  private func addPendingFrames(_ delta: Int64) {
    framesLock.lock()
    pendingPlaybackFrames = max(0, pendingPlaybackFrames + delta)
    framesLock.unlock()
  }

  private func setPendingFrames(_ value: Int64) {
    framesLock.lock()
    pendingPlaybackFrames = value
    framesLock.unlock()
  }

  private func stopAudio() {
    isCapturing = false
    isPlaying = false
    engine.inputNode.removeTap(onBus: 0) // unconditional: safe no-op if no tap installed
    if engine.isRunning {
      player.stop()
      engine.stop()
    }
    engine.reset() // note: does NOT detach nodes — player stays attached across sessions by design
    captureConverter = nil
    captureMuted = false
    setPendingFrames(0)
    try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
  }
}
