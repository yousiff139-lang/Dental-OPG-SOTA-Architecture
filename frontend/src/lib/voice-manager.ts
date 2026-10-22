"use client";

// Clean text for natural speech synthesis
export function cleanTextForSpeech(text: string): string {
  return text
    .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "") // Remove emojis
    .replace(/[*_#`~]/g, "") // Remove markdown formatting
    .replace(/\bXRayBo\b/gi, "X-Ray Bo") // Pronounce correctly
    .replace(/\s+/g, " ")
    .trim();
}

export interface VoicePlaybackCallbacks {
  onStart?: (info: { duration: number }) => void;
  onEnd?: () => void;
  onError?: (err: unknown) => void;
}

class VoiceManager {
  private audioElement: HTMLAudioElement | null = null;
  private isMuted: boolean = false;
  private isSpeaking: boolean = false;
  private listeners: Set<(isSpeaking: boolean) => void> = new Set();
  private blobCache: Map<string, string> = new Map();
  private isPreloading: boolean = false;

  constructor() {
    if (typeof window !== "undefined") {
      this.initAudioElement();
    }
  }

  private initAudioElement() {
    if (typeof window === "undefined") return;
    if (!this.audioElement) {
      this.audioElement = new Audio();
      this.audioElement.preload = "auto";
    }
  }

  public subscribe(listener: (isSpeaking: boolean) => void): () => void {
    this.listeners.add(listener);
    listener(this.isSpeaking);
    return () => this.listeners.delete(listener);
  }

  private notify(isSpeaking: boolean) {
    this.isSpeaking = isSpeaking;
    this.listeners.forEach((cb) => cb(isSpeaking));
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted) {
      this.stop();
    }
  }

  public getMuted(): boolean {
    return this.isMuted;
  }

  public stop() {
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement.currentTime = 0;
    }
    this.notify(false);
  }

  // Pre-load standard messages in the background so audio playback starts with 0ms delay
  public async preloadMessages(messages: string[]) {
    if (this.isPreloading || typeof window === "undefined") return;
    this.isPreloading = true;

    for (const msg of messages) {
      const clean = cleanTextForSpeech(msg);
      if (!clean || this.blobCache.has(clean)) continue;

      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
        const res = await fetch(`${apiUrl}/tts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: clean }),
        });
        if (res.ok) {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          this.blobCache.set(clean, url);
        }
      } catch {
        // Silently continue preloading
      }
    }
    this.isPreloading = false;
  }

  // Prime and unlock audio during a user click event (like clicking Accept on modal)
  public unlockAudioContext() {
    if (typeof window === "undefined") return;

    try {
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (AudioContextClass) {
        const ctx = new AudioContextClass();
        if (ctx.state === "suspended") {
          ctx.resume();
        }
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        gain.gain.value = 0.001;
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.05);
      }
    } catch {
      // Ignore
    }

    this.initAudioElement();
  }

  // Fetch audio from Fish Audio backend or local cache
  public async getAudioUrl(text: string, fallbackPath?: string): Promise<string | null> {
    const cleanText = cleanTextForSpeech(text);
    if (!cleanText) return null;

    // Check in-memory blob cache
    if (this.blobCache.has(cleanText)) {
      return this.blobCache.get(cleanText)!;
    }

    // Call Fish Audio TTS backend
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
      const response = await fetch(`${apiUrl}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanText }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        this.blobCache.set(cleanText, audioUrl);
        return audioUrl;
      }
    } catch (err) {
      console.warn("Fish Audio backend call failed:", err);
    }

    // Return fallback local path if available
    return fallbackPath || null;
  }

  /**
   * Main method to speak using Fish Audio.
   * Calls onStart the exact millisecond audio playback starts, allowing
   * the UI bubble and typewriter to synchronize in lockstep with the voice.
   */
  public async speakMessage(
    text: string,
    options?: VoicePlaybackCallbacks & { fallbackPath?: string }
  ): Promise<boolean> {
    if (this.isMuted) return false;
    this.stop();

    const cleanText = cleanTextForSpeech(text);
    if (!cleanText) return false;

    this.initAudioElement();
    if (!this.audioElement) return false;

    try {
      const audioUrl = await this.getAudioUrl(cleanText, options?.fallbackPath);
      if (!audioUrl) {
        options?.onError?.(new Error("Failed to obtain audio for message"));
        return false;
      }

      this.audioElement.src = audioUrl;

      // Event: playback actually begins sound emission
      this.audioElement.onplay = () => {
        this.notify(true);
        const duration = this.audioElement?.duration || 3.5;
        options?.onStart?.({ duration });
      };

      // Event: playback completes
      this.audioElement.onended = () => {
        this.notify(false);
        options?.onEnd?.();
      };

      // Event: playback error
      this.audioElement.onerror = (e) => {
        console.warn("Audio element error during playback:", e);
        this.notify(false);
        options?.onError?.(e);
      };

      await this.audioElement.play();
      return true;
    } catch (err) {
      console.warn("Error playing Fish Audio:", err);
      this.notify(false);
      options?.onError?.(err);
      return false;
    }
  }
}

export const voiceManager = new VoiceManager();
