// Voice I/O for the chat UI. Everything stays local: microphone audio goes to
// OUR backend over the same /ws/chat socket, and reply audio comes back from
// our local Piper TTS. No browser speech APIs are used.
//
// app.js creates one instance:  const voice = createVoice({ getSocket, getSessionId, onRecordingChange, onError });
window.createVoice = function (opts) {
  let ctx = null; // one AudioContext for both capture and playback
  let workletLoaded = false;
  let micStream = null;
  let sourceNode = null;
  let captureNode = null;
  let recording = false;

  // Playback state (reset at the start of every turn)
  let expectedSeq = 0; // next chunk we are allowed to play
  let pending = {}; // chunks that arrived early, keyed by seq
  let nextStartTime = 0; // where the next chunk starts on the audio clock
  let playing = []; // scheduled AudioBufferSourceNodes (for Stop)
  let stoppedThisTurn = false;

  async function ensureContext() {
    if (!ctx) ctx = new AudioContext();
    if (ctx.state === "suspended") await ctx.resume();
    return ctx;
  }

  // ---------------------------------------------------------------- capture
  // speak = whether the server should synthesise the reply as audio.
  async function startMic(speak) {
    if (recording) return;
    const socket = opts.getSocket();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      opts.onError("Not connected. Reconnect and try again.");
      return;
    }
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      opts.onError(
        err && err.name === "NotAllowedError"
          ? "Microphone permission denied. Click the lock icon in the address bar, allow the microphone, then try again."
          : "Could not open the microphone: " + (err && err.message ? err.message : err)
      );
      return;
    }
    await ensureContext();
    if (!workletLoaded) {
      await ctx.audioWorklet.addModule("/assets/audio-worklet.js");
      workletLoaded = true;
    }
    sourceNode = ctx.createMediaStreamSource(micStream);
    captureNode = new AudioWorkletNode(ctx, "pcm-capture");
    captureNode.port.onmessage = function (e) {
      const s = opts.getSocket();
      if (recording && s && s.readyState === WebSocket.OPEN) s.send(e.data); // binary frame
    };
    // Connect through a silent gain so the browser keeps processing the node.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    sourceNode.connect(captureNode).connect(mute).connect(ctx.destination);

    resetPlayback();
    socket.send(
      JSON.stringify({
        type: "audio_start",
        session_id: opts.getSessionId(),
        sample_rate: 16000,
        format: "pcm16",
        speak: speak !== false,
      })
    );
    recording = true;
    opts.onRecordingChange(true);
  }

  // sendStop=true when the user clicks the button; false when the server
  // already detected the end of speech.
  function stopMic(sendStop) {
    if (!recording) return;
    recording = false;
    const socket = opts.getSocket();
    if (sendStop && socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "audio_stop", session_id: opts.getSessionId() }));
    }
    if (sourceNode) sourceNode.disconnect();
    if (captureNode) captureNode.disconnect();
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
    sourceNode = captureNode = micStream = null;
    opts.onRecordingChange(false);
  }

  // --------------------------------------------------------------- playback
  function resetPlayback() {
    expectedSeq = 0;
    pending = {};
    stoppedThisTurn = false;
  }

  // Chunks are played strictly in seq order, back to back on the audio clock,
  // so there are no gaps or overlaps between sentences.
  async function playChunk(msg) {
    if (stoppedThisTurn) return;
    await ensureContext();
    pending[msg.seq] = msg;
    while (pending[expectedSeq]) {
      schedule(pending[expectedSeq]);
      delete pending[expectedSeq];
      expectedSeq += 1;
    }
  }

  function schedule(msg) {
    const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer);
    const buffer = ctx.createBuffer(1, pcm.length, msg.sample_rate);
    const out = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;

    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.03, nextStartTime);
    node.start(startAt);
    nextStartTime = startAt + buffer.duration;
    playing.push(node);
    node.onended = function () {
      playing = playing.filter((n) => n !== node);
    };
  }

  function stopPlayback() {
    stoppedThisTurn = true; // ignore the rest of this reply's audio
    playing.forEach((n) => {
      try {
        n.stop();
      } catch (e) {
        /* already stopped */
      }
    });
    playing = [];
    pending = {};
    nextStartTime = 0;
  }

  function isPlaying() {
    return playing.length > 0;
  }

  return { startMic, stopMic, isRecording: () => recording, playChunk, resetPlayback, stopPlayback, isPlaying };
};
