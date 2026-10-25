// AudioWorklet processor: runs on the browser's audio thread.
// Takes microphone audio at the device rate (usually 48 kHz), downsamples it
// to 16 kHz mono PCM16 by averaging, and posts 512-sample (32 ms) frames to
// the main thread, which sends them to the backend as binary WebSocket frames.
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // input samples per output sample
    this.pos = 0; // how far we are into the current output sample's window
    this.sum = 0;
    this.count = 0;
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0][0]; // first channel = mono
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        // One output sample = average of the input samples in its window.
        this.pos -= this.ratio;
        const s = Math.max(-1, Math.min(1, this.sum / this.count));
        this.frame[this.filled++] = s * 32767;
        this.sum = 0;
        this.count = 0;
        if (this.filled === FRAME_SAMPLES) {
          this.port.postMessage(this.frame.buffer.slice(0)); // copy, then reuse
          this.filled = 0;
        }
      }
    }
    return true; // keep the processor alive
  }
}

registerProcessor("pcm-capture", PcmCapture);
