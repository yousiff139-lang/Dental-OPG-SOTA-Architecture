"use client";

/**
 * ARCHIVED / LEGACY ROBOTIC VOICE MODULE (Web Speech API)
 * Kept in this separate file for potential future use as requested.
 * This file is completely detached from the active Fish Audio system.
 */

export class LegacyRoboticVoice {
  private isMuted: boolean = false;

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted && typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  public speak(text: string, onEnd?: () => void) {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (this.isMuted) return;

    window.speechSynthesis.cancel();

    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }

    const cleanText = text
      .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "")
      .replace(/[*_#`~]/g, "")
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const voices = window.speechSynthesis.getVoices();
    const defaultVoice = voices.find((v) => v.lang.startsWith("en")) || voices[0] || null;

    if (defaultVoice) {
      utterance.voice = defaultVoice;
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    if (onEnd) {
      utterance.onend = onEnd;
      utterance.onerror = onEnd;
    }

    window.speechSynthesis.speak(utterance);
  }

  public stop() {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }
}

export const legacyRoboticVoice = new LegacyRoboticVoice();
