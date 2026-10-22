"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Bot, Volume2, VolumeX, Sparkles, RefreshCw } from "lucide-react";
import { voiceManager } from "@/lib/voice-manager";

export type RobotState =
    | "idle"
    | "file_uploaded"
    | "predicting"
    | "prediction_complete"
    | "generating_report"
    | "report_complete";

interface RobotChatBubbleProps {
    state: RobotState;
    fileName?: string;
    conditionsFound?: number;
    isRobotLoaded?: boolean;
    hasConsented?: boolean;
}

const MESSAGES: Record<RobotState, string> = {
    idle: "👋 Hey there! Welcome to XRayBo! Upload a dental X-ray image and I'll help you detect cavities and other conditions!",
    file_uploaded: "📸 Great! Your X-ray is ready. Click the **Predict** button and I'll analyze it for cavities and lesions!",
    predicting: "🔍 Analyzing your X-ray... Scanning for cavities, periapical lesions, and tooth conditions...",
    prediction_complete: "✅ Analysis complete! I found some dental findings. Click **Generate Diagnosis Report** for detailed medical insights!",
    generating_report: "📝 Generating your comprehensive dental diagnosis report with treatment recommendations...",
    report_complete: "🎉 Your diagnosis report is ready! Review the detailed findings, severity levels, and clinical recommendations below.",
};

// Gentle pleasant chime when chat bubble appears
function playPopChime() {
    try {
        const AudioContextClass =
            window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioContextClass) return;
        const ctx = new AudioContextClass();
        if (ctx.state === "suspended") {
            ctx.resume();
        }
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
        osc.frequency.exponentialRampToValueAtTime(783.99, ctx.currentTime + 0.07); // G5
        gain.gain.setValueAtTime(0.02, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.1);
    } catch {
        // Audio context blocked or not ready
    }
}

export function RobotChatBubble({
    state,
    conditionsFound,
    hasConsented = false
}: RobotChatBubbleProps) {
    const [isVisible, setIsVisible] = useState(false);
    const [displayedText, setDisplayedText] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);

    const typewriterTimerRef = useRef<NodeJS.Timeout | null>(null);
    const lastSpokenStateRef = useRef<string | null>(null);

    // Subscribe to voiceManager speaking status
    useEffect(() => {
        const unsubscribe = voiceManager.subscribe((speaking) => {
            setIsSpeaking(speaking);
        });
        return () => unsubscribe();
    }, []);

    // Preload all standard messages on mount so transitions are instant
    useEffect(() => {
        if (hasConsented) {
            voiceManager.preloadMessages(Object.values(MESSAGES));
        }
    }, [hasConsented]);

    // Synchronized typewriter typing
    const startSynchronizedTypewriter = useCallback((fullText: string, speedMs: number = 20) => {
        if (typewriterTimerRef.current) {
            clearInterval(typewriterTimerRef.current);
            typewriterTimerRef.current = null;
        }

        setIsTyping(true);
        setDisplayedText("");
        let index = 0;

        const interval = setInterval(() => {
            if (index < fullText.length) {
                setDisplayedText(fullText.slice(0, index + 1));
                index++;
            } else {
                setIsTyping(false);
                clearInterval(interval);
                typewriterTimerRef.current = null;
            }
        }, speedMs);

        typewriterTimerRef.current = interval;
    }, []);

    // Get current message text
    const getMessageText = useCallback(
        (targetState: RobotState, numConditions?: number): string => {
            if (targetState === "prediction_complete" && numConditions !== undefined) {
                return `✅ Analysis complete! I found **${numConditions} condition${numConditions !== 1 ? "s" : ""}**. Click **Generate Diagnosis Report** for detailed recommendations!`;
            }
            return MESSAGES[targetState] || MESSAGES.idle;
        },
        []
    );

    // Determine audio asset path if pre-recorded fallback
    const getAudioAssetPath = useCallback(
        (targetState: RobotState, numConditions?: number): string => {
            let assetKey: string = targetState;
            if (targetState === "prediction_complete" && numConditions !== undefined) {
                if (numConditions >= 0 && numConditions <= 4) {
                    assetKey = `prediction_complete_${numConditions}`;
                } else {
                    assetKey = "prediction_complete";
                }
            }
            return `/audio/calm/${assetKey}.mp3`;
        },
        []
    );

    // Synchronized playback: Bubble pops up and types in lockstep with the voice
    const speakCurrentState = useCallback(
        (targetState: RobotState, numConditions?: number, force = false) => {
            const message = getMessageText(targetState, numConditions);
            const fallbackAudioPath = getAudioAssetPath(targetState, numConditions);

            lastSpokenStateRef.current = targetState;

            // Trigger Fish Audio with synchronization callbacks
            voiceManager.speakMessage(message, {
                fallbackPath: fallbackAudioPath,
                onStart: ({ duration }) => {
                    // Synchronized: Pop up bubble and start typewriter at the exact moment voice speaks
                    setIsVisible(true);
                    playPopChime();
                    const typingSpeed = duration > 0
                        ? Math.max(12, Math.floor((duration * 1000) / message.length))
                        : 20;
                    startSynchronizedTypewriter(message, typingSpeed);
                },
                onError: () => {
                    // Fallback visual display if voice is interrupted
                    setIsVisible(true);
                    startSynchronizedTypewriter(message, 20);
                },
            });

            // If muted, show visual typing immediately
            if (isMuted && !force) {
                setIsVisible(true);
                startSynchronizedTypewriter(message, 20);
            }
        },
        [getMessageText, getAudioAssetPath, isMuted, startSynchronizedTypewriter]
    );

    // Triggers on state change or consent granted
    useEffect(() => {
        if (!hasConsented) {
            setIsVisible(false);
            return;
        }

        if (lastSpokenStateRef.current !== state) {
            speakCurrentState(state, conditionsFound);
        }
    }, [state, conditionsFound, hasConsented, speakCurrentState]);

    // Manual Replay / Read message out loud
    const handleReplayVoice = () => {
        voiceManager.unlockAudioContext();
        speakCurrentState(state, conditionsFound, true);
    };

    // Toggle mute
    const toggleMute = () => {
        const nextMuted = !isMuted;
        setIsMuted(nextMuted);
        voiceManager.setMuted(nextMuted);
    };

    // If user hasn't consented yet, don't show the bubble
    if (!hasConsented) {
        return null;
    }

    return (
        <div
            className={`
        absolute top-4 left-4 right-4 z-30
        transition-all duration-300 ease-out
        ${isVisible ? "opacity-100 translate-y-0 scale-100" : "opacity-0 -translate-y-2 scale-95 pointer-events-none"}
      `}
        >
            {/* Chat Bubble Container */}
            <div className="relative">
                {/* Glassmorphism bubble */}
                <div className="
          bg-black/80 backdrop-blur-xl
          border border-white/20
          rounded-2xl
          p-4
          shadow-2xl
          shadow-primary/25
        ">
                    {/* Header Row */}
                    <div className="flex items-start gap-3">
                        <div className="shrink-0 p-2 rounded-full bg-primary/20 border border-primary/30 animate-pulse">
                            <Bot className="size-5 text-primary" />
                        </div>

                        {/* Content & Controls */}
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <p className="text-xs font-semibold text-primary">AI Assistant</p>
                                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-medium inline-flex items-center gap-1">
                                        <Sparkles className="size-2.5" />
                                        Fish Audio (Calm)
                                    </span>
                                    {isSpeaking && (
                                        <span className="inline-flex items-center gap-1 text-[10px] text-primary/90 font-medium animate-pulse">
                                            <span className="size-1.5 rounded-full bg-primary animate-ping" />
                                            Speaking...
                                        </span>
                                    )}
                                </div>

                                {/* Controls */}
                                <div className="flex items-center gap-1.5">
                                    {/* Replay / Read message button */}
                                    <button
                                        onClick={handleReplayVoice}
                                        className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-primary/20 hover:bg-primary/30 border border-primary/40 text-[11px] text-primary transition-all cursor-pointer"
                                        title="Click to hear voice read this message"
                                        aria-label="Read message aloud"
                                    >
                                        <RefreshCw className={`size-3 ${isSpeaking ? "animate-spin" : ""}`} />
                                        <span>Read Aloud</span>
                                    </button>

                                    {/* Mute Control */}
                                    <button
                                        onClick={toggleMute}
                                        className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 border border-white/10 transition-colors cursor-pointer"
                                        title={isMuted ? "Unmute voice" : "Mute voice"}
                                        aria-label={isMuted ? "Unmute voice" : "Mute voice"}
                                    >
                                        {isMuted ? (
                                            <VolumeX className="size-3.5 text-white/50" />
                                        ) : (
                                            <Volume2 className="size-3.5 text-primary" />
                                        )}
                                    </button>
                                </div>
                            </div>

                            {/* Message Display */}
                            <p className="text-sm text-white/95 leading-relaxed whitespace-pre-wrap font-sans">
                                {displayedText}
                                {isTyping && (
                                    <span className="inline-block w-1 h-3.5 ml-0.5 bg-primary animate-pulse align-middle" />
                                )}
                            </p>
                        </div>
                    </div>

                    {/* Loading indicator for processing states */}
                    {(state === "predicting" || state === "generating_report") && (
                        <div className="mt-3 flex items-center gap-2 pt-2 border-t border-white/10">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
                                <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
                                <span className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
                            </div>
                            <span className="text-xs text-white/60">
                                {state === "predicting" ? "Analyzing dental X-ray image..." : "Generating diagnostic report..."}
                            </span>
                        </div>
                    )}
                </div>

                {/* Speech bubble tail */}
                <div className="absolute -bottom-2 right-8 w-4 h-4 bg-black/80 backdrop-blur-xl border-r border-b border-white/20 rotate-45 transform" />
            </div>
        </div>
    );
}
