import { NextRequest, NextResponse } from "next/server";

// In-memory audio cache to reduce API calls and eliminate latency for repeated phrases
const audioCache = new Map<string, ArrayBuffer>();

function cleanTextForSpeech(text: string): string {
  return text
    .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "")
    .replace(/[*_#`~]/g, "")
    .replace(/\bXRayBo\b/gi, "X-Ray Bo")
    .replace(/\s+/g, " ")
    .trim();
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const rawText = body.text;

    if (!rawText || typeof rawText !== "string") {
      return NextResponse.json({ error: "Missing or invalid text parameter" }, { status: 400 });
    }

    const cleanText = cleanTextForSpeech(rawText);
    if (!cleanText) {
      return NextResponse.json({ error: "Text is empty after cleaning" }, { status: 400 });
    }

    const apiKey =
      process.env.FISH_AUDIO_API_KEY || "sk-fish-CDQEhTkpJKgzGigPhzumuWzIYIz37l7DPh1W_Fbynsg";

    // Check cache
    const cacheKey = cleanText;
    if (audioCache.has(cacheKey)) {
      const cached = audioCache.get(cacheKey)!;
      return new Response(cached, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "Cache-Control": "public, max-age=86400",
        },
      });
    }

    // Call Fish Audio API with Calm Normal Human Voice model
    const fishResponse = await fetch("https://api.fish.audio/v1/tts", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        model: "s2.1-pro-free",
      },
      body: JSON.stringify({
        text: cleanText,
        format: "mp3",
        reference_id: "0b74ead073f2474a904f69033535b98e", // Calm Normal Voice
        latency: "normal",
      }),
    });

    if (!fishResponse.ok) {
      const errText = await fishResponse.text();
      console.error("Fish Audio API error:", fishResponse.status, errText);
      return NextResponse.json(
        { error: "Fish Audio API failed", status: fishResponse.status, details: errText },
        { status: fishResponse.status }
      );
    }

    const audioArrayBuffer = await fishResponse.arrayBuffer();

    // Cache result
    if (audioCache.size > 50) {
      const firstKey = audioCache.keys().next().value;
      if (firstKey) audioCache.delete(firstKey);
    }
    audioCache.set(cacheKey, audioArrayBuffer);

    return new Response(audioArrayBuffer, {
      status: 200,
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    const stack = err instanceof Error ? err.stack : undefined;
    console.error("TTS Route Exception:", message, stack);
    return NextResponse.json(
      { error: "Internal Server Error", message, stack },
      { status: 500 }
    );
  }
}
