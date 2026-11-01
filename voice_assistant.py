"""
Voice assistant for the Glacial Lake Monitoring app.
----------------------------------------------------
Same idea as the chat assistant, but it is SEPARATE code with its own tool:

    microphone audio --(Groq Whisper)--> text question
    text question    --(LangChain agent + voice_result_lookup tool)--> short answer
    answer text      --(gTTS, optional)--> spoken audio

Uses the same GROQ_API_KEY as the chat assistant. `gTTS` is optional: if it
is not installed (or has no internet), the answer is simply shown as text.
"""
import io
import os

from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
STT_LANGUAGE = os.environ.get("GROQ_STT_LANGUAGE", "en")   # "en" = English, "ta" = Tamil


# ----------------------------------------------------------------------------
# 1. Speech -> text
# ----------------------------------------------------------------------------
def transcribe(audio_bytes):
    """Turn recorded audio (WAV bytes) into text using Groq Whisper."""
    from groq import Groq          # installed together with langchain-groq
    result = Groq().audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model=STT_MODEL,
        language=STT_LANGUAGE,      # fixed language, so Whisper does not guess wrongly
        response_format="verbose_json",   # also tells us how likely each part is silence
    )

    def field(seg, name):
        return seg.get(name) if isinstance(seg, dict) else getattr(seg, name, None)

    # Whisper can "hear" words in silence, so drop parts it marks as no speech
    segments = getattr(result, "segments", None) or []
    if segments:
        kept = [field(s, "text") or "" for s in segments if (field(s, "no_speech_prob") or 0) < 0.5]
        text = " ".join(kept).strip()
    else:
        text = result.text.strip()

    # typical phrases Whisper makes up when there is only silence or noise
    made_up = {"thank you", "thank you.", "thanks for watching", "thanks for watching!", "you", "bye", "bye."}
    if text.lower() in made_up or not any(ch.isalnum() for ch in text):
        return ""
    return text


# ----------------------------------------------------------------------------
# 2. LangChain agent (Groq LLM + its own tool)
# ----------------------------------------------------------------------------
def build_voice_agent(facts, subject):
    """Voice agent = Groq LLM + the voice_result_lookup tool."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def voice_result_lookup(item: str) -> str:
        """Look up the measured results of the satellite image comparison.
        item must be one of: area_first, area_second, unchanged, new_area,
        lost_area, change_percent, status, risk, risk_score.
        Use item='all' to get every measurement at once."""
        key = item.strip().lower().replace(" ", "_")
        if key == "all":
            return "\n".join(f"{k}: {v}" for k, v in facts.items())
        return facts.get(key, f"Unknown item '{item}'. Valid items: all, {', '.join(facts)}")

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)      # reads GROQ_API_KEY itself
    return create_agent(
        llm,
        tools=[voice_result_lookup],
        system_prompt=(
            f"You are a voice assistant explaining the result of comparing two satellite "
            f"images of a {subject.lower()}. Always use the voice_result_lookup tool to get "
            "numbers. Never guess or invent values. Areas are shares of the image, not "
            "square kilometres. Your answer will be read aloud: reply in one to three short "
            "plain sentences, with no markdown, bullet points or symbols."
        ),
    )


def ask_voice(agent, question):
    """Send one spoken question (as text) to the agent and return its answer."""
    out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return out["messages"][-1].content


# ----------------------------------------------------------------------------
# 3. Text -> speech (optional)
# ----------------------------------------------------------------------------
def speak(text):
    """Answer text -> MP3 bytes, or None if gTTS is missing / offline."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 4. One call for the app: audio in -> (question, answer, answer_audio)
# ----------------------------------------------------------------------------
def answer_audio(audio_bytes, facts, subject):
    question = transcribe(audio_bytes)
    if not question:
        return "", "", None        # nothing was said
    answer = ask_voice(build_voice_agent(facts, subject), question)
    return question, answer, speak(answer)
