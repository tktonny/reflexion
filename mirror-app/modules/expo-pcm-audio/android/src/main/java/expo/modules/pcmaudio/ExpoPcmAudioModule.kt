package expo.modules.pcmaudio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Bundle
import android.util.Base64
import expo.modules.kotlin.Promise
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Streaming PCM bridge for v3 (direct Qwen realtime WS).
 *
 *  - Capture: mic -> PCM16 mono @ 16 kHz, ~100 ms frames, base64, emitted as `onAudioChunk`.
 *    Uses VOICE_COMMUNICATION source + platform AEC/NS/AGC so the assistant's own voice from the
 *    speaker is not fed back into the mic (important for a hands-free smart mirror).
 *  - Playback: base64 PCM16 mono @ 24 kHz enqueued via play(); a single writer thread drains the
 *    queue into an AudioTrack (MODE_STREAM) so rapid deltas stay ordered and never block JS.
 *  - Half-duplex: setCaptureMuted(true) during assistant playback drops captured frames.
 */
class ExpoPcmAudioModule : Module() {
  // ---- capture (mic -> upstream) ----
  private var recorder: AudioRecord? = null
  private var captureThread: Thread? = null
  private val capturing = AtomicBoolean(false)
  private val captureMuted = AtomicBoolean(false)
  private var captureSampleRate = 16000
  private var aec: AcousticEchoCanceler? = null
  private var ns: NoiseSuppressor? = null
  private var agc: AutomaticGainControl? = null

  // ---- playback (downstream deltas -> speaker) ----
  private var track: AudioTrack? = null
  private var playbackThread: Thread? = null
  private val playing = AtomicBoolean(false)
  private val playbackQueue = LinkedBlockingQueue<ByteArray>()
  private val playbackSampleRate = 24000
  // Bytes handed to AudioTrack.write() so far; vs playbackHeadPosition it yields the unplayed
  // backlog, so the JS side un-mutes the mic only after playback has truly drained (half-duplex).
  private val totalBytesWritten = AtomicLong(0)

  override fun definition() = ModuleDefinition {
    Name("ExpoPcmAudio")

    Events("onAudioChunk")

    AsyncFunction("start") { sampleRate: Int, promise: Promise ->
      try {
        captureSampleRate = if (sampleRate > 0) sampleRate else 16000
        startPlayback()
        startCapture()
        promise.resolve(null)
      } catch (e: Throwable) {
        stopCaptureInternal()
        stopPlaybackInternal()
        promise.reject("ERR_PCM_START", e.message ?: "failed to start PCM audio", e)
      }
    }

    AsyncFunction("stop") { promise: Promise ->
      stopCaptureInternal()
      stopPlaybackInternal()
      promise.resolve(null)
    }

    // Cheap + non-blocking: decode base64 and enqueue. The writer thread does the blocking write.
    Function("play") { base64: String ->
      if (!playing.get()) return@Function
      val bytes = try { Base64.decode(base64, Base64.DEFAULT) } catch (_: Throwable) { return@Function }
      if (bytes.isNotEmpty()) playbackQueue.offer(bytes)
    }

    Function("clearPlayback") {
      playbackQueue.clear()
      totalBytesWritten.set(0) // flush() also resets playbackHeadPosition to 0 — keep them in sync
      track?.let { try { it.pause(); it.flush(); it.play() } catch (_: Throwable) {} }
    }

    Function("setCaptureMuted") { muted: Boolean ->
      captureMuted.set(muted)
    }

    // Unplayed playback backlog in ms (queued + buffered but not yet rendered). The JS hook polls
    // this so it un-mutes the mic only once the assistant has actually stopped talking.
    Function("getPlaybackBacklogMs") {
      val t = track
      if (!playing.get() || t == null) return@Function 0.0
      val playedBytes = (t.playbackHeadPosition.toLong() and 0xFFFFFFFFL) * 2 // frames -> bytes (mono 16-bit)
      val writtenBacklog = (totalBytesWritten.get() - playedBytes).coerceAtLeast(0)
      // Bytes still in the queue, not yet written to the track. This is the crucial term: at
      // response.done ALL deltas are enqueued, but the writer thread has only pushed ~0.5 s into
      // AudioTrack, so writtenBacklog alone reads near-zero — the mic would re-open while seconds
      // of the assistant's speech are still queued, and the speaker would feed a whole sentence
      // back into the mic (server-VAD then transcribes the echo as a user turn). Count the queue.
      var queuedBytes = 0L
      for (chunk in playbackQueue) queuedBytes += chunk.size.toLong()
      val backlogBytes = writtenBacklog + queuedBytes
      backlogBytes.toDouble() / (playbackSampleRate * 2) * 1000.0
    }

    OnDestroy {
      stopCaptureInternal()
      stopPlaybackInternal()
    }
  }

  private fun startCapture() {
    if (capturing.get()) return
    val minBuf = AudioRecord.getMinBufferSize(
      captureSampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
    )
    if (minBuf <= 0) throw IllegalStateException("mic unavailable (minBufferSize=$minBuf)")
    // ~100 ms per read: 16000 samples/s / 10 * 2 bytes = 3200 bytes; never below the platform min.
    val chunkBytes = maxOf(minBuf, captureSampleRate / 10 * 2)
    val rec = AudioRecord(
      MediaRecorder.AudioSource.VOICE_COMMUNICATION, // platform echo-cancel path for two-way voice
      captureSampleRate,
      AudioFormat.CHANNEL_IN_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
      chunkBytes * 4,
    )
    if (rec.state != AudioRecord.STATE_INITIALIZED) {
      rec.release()
      throw IllegalStateException("AudioRecord failed to initialize (mic permission?)")
    }
    enableEffects(rec.audioSessionId)
    recorder = rec
    capturing.set(true)
    rec.startRecording()
    captureThread = Thread {
      val buf = ByteArray(chunkBytes)
      while (capturing.get()) {
        val n = rec.read(buf, 0, buf.size)
        if (n > 0) {
          // Always read (drain the mic); only emit when not muted (half-duplex).
          if (!captureMuted.get()) {
            val b64 = Base64.encodeToString(buf, 0, n, Base64.NO_WRAP)
            sendEvent("onAudioChunk", Bundle().apply { putString("data", b64) })
          }
        } else if (n < 0) {
          break
        }
      }
    }.apply { name = "pcm-capture"; isDaemon = true; start() }
  }

  private fun enableEffects(sessionId: Int) {
    try { if (AcousticEchoCanceler.isAvailable()) aec = AcousticEchoCanceler.create(sessionId)?.apply { enabled = true } } catch (_: Throwable) {}
    try { if (NoiseSuppressor.isAvailable()) ns = NoiseSuppressor.create(sessionId)?.apply { enabled = true } } catch (_: Throwable) {}
    try { if (AutomaticGainControl.isAvailable()) agc = AutomaticGainControl.create(sessionId)?.apply { enabled = true } } catch (_: Throwable) {}
  }

  private fun stopCaptureInternal() {
    capturing.set(false)
    try { captureThread?.join(500) } catch (_: Throwable) {}
    captureThread = null
    try { recorder?.stop() } catch (_: Throwable) {}
    try { recorder?.release() } catch (_: Throwable) {}
    recorder = null
    try { aec?.release() } catch (_: Throwable) {}
    aec = null
    try { ns?.release() } catch (_: Throwable) {}
    ns = null
    try { agc?.release() } catch (_: Throwable) {}
    agc = null
    captureMuted.set(false)
  }

  private fun buildTrack(): AudioTrack {
    val minBuf = AudioTrack.getMinBufferSize(
      playbackSampleRate,
      AudioFormat.CHANNEL_OUT_MONO,
      AudioFormat.ENCODING_PCM_16BIT,
    )
    if (minBuf <= 0) throw IllegalStateException("audio output unavailable (minBufferSize=$minBuf)")
    val bufSize = maxOf(minBuf, playbackSampleRate) // ~0.5 s headroom against underrun
    val t = AudioTrack(
      AudioAttributes.Builder()
        // USAGE_MEDIA (not VOICE_COMMUNICATION): the media path reliably routes to the mirror's
        // main loudspeaker at an audible, user-controllable level without MODE_IN_COMMUNICATION.
        // Feedback is prevented by half-duplex muting, not by the voice-call route.
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build(),
      AudioFormat.Builder()
        .setSampleRate(playbackSampleRate)
        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build(),
      bufSize,
      AudioTrack.MODE_STREAM,
      AudioManager.AUDIO_SESSION_ID_GENERATE,
    )
    if (t.state != AudioTrack.STATE_INITIALIZED) {
      t.release()
      throw IllegalStateException("AudioTrack failed to initialize")
    }
    return t
  }

  private fun startPlayback() {
    if (playing.get()) return
    val t = buildTrack()
    track = t
    playing.set(true)
    playbackQueue.clear()
    totalBytesWritten.set(0)
    t.play()
    playbackThread = Thread {
      while (playing.get()) {
        val bytes = try {
          playbackQueue.poll(100, TimeUnit.MILLISECONDS)
        } catch (_: InterruptedException) {
          null
        } ?: continue
        var off = 0
        while (off < bytes.size && playing.get()) {
          val cur = track ?: break
          val w = cur.write(bytes, off, bytes.size - off)
          when {
            w >= 0 -> {
              off += w
              totalBytesWritten.addAndGet(w.toLong())
            }
            w == AudioTrack.ERROR_DEAD_OBJECT -> {
              // audioserver restarted: rebuild the track and retry the remainder of this chunk.
              try {
                cur.release()
                val fresh = buildTrack()
                fresh.play()
                track = fresh
                totalBytesWritten.set(0)
              } catch (_: Throwable) {
                playing.set(false)
                break
              }
            }
            else -> break // ERROR_INVALID_OPERATION / ERROR_BAD_VALUE: drop this chunk, keep track
          }
        }
      }
    }.apply { name = "pcm-playback"; isDaemon = true; start() }
  }

  private fun stopPlaybackInternal() {
    playing.set(false)
    playbackQueue.clear()
    try { playbackThread?.join(500) } catch (_: Throwable) {}
    playbackThread = null
    try { track?.pause(); track?.flush(); track?.stop() } catch (_: Throwable) {}
    try { track?.release() } catch (_: Throwable) {}
    track = null
  }
}
