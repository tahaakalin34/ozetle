import json
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI(title="Özetle API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# GÜVENLİ YÖNTEM: Anahtar artık kodda değil, Vercel'in gizli kasasından okunuyor!
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """Sen uzman bir eğitim tasarımcısı ve bilgi editörüsün. Sana [mm:ss] zaman damgalı bir video transkripti verilecek. Onu mikro-öğrenme paketine dönüştür.

KURALLAR:
1. YALNIZCA geçerli JSON döndür. Markdown, açıklama veya kod bloğu (```json) kullanma.
2. Sadece transkriptte geçen bilgiyi kullan, uydurma.
3. Çıktı dili: {lang}.
4. cards: 6-10 adet, konuşmanın akışını izlesin. ts, ilgili anın başlangıcı (tamsayı saniye). Başlık en fazla 8 kelime ve vurucu olsun. Her kart için üç derinlik yaz: eli5 (12 yaşındaki biri anlasın), standard (2-3 cümle), expert (terim ve nüans içeren akademik/teknik anlatım).
5. quotes: konuşmacının gerçekten söylediği 3-5 vurucu cümle, kelimesi kelimesine, ts ile.
6. actions: 3-5 somut, ölçülebilir, hemen uygulanabilir eylem.
7. quiz: 3-5 çoktan seçmeli soru, 4 şık. answer doğru şıkkın indeksi (0-3), why kısa gerekçe.
8. concept_graph: 5-8 düğüm (type: idea|example|result). Kenarlardaki from/to düğüm id'sidir.

ŞEMA:
{"meta":{"title":str,"speaker":str},
 "big_idea":str,"tldr":[str],
 "cards":[{"ts":int,"icon":emoji,"title":str,"eli5":str,"standard":str,"expert":str}],
 "quotes":[{"text":str,"ts":int}],
 "actions":[str],
 "quiz":[{"q":str,"options":[str,str,str,str],"answer":int,"why":str}],
 "concept_graph":{"nodes":[{"id":str,"label":str,"type":str}],"edges":[{"from":str,"to":str,"label":str}]}}"""


def extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", url)
    if not m:
        raise HTTPException(status_code=400, detail="Geçerli bir YouTube bağlantısı girin.")
    return m.group(1)


def fetch_transcript_text(vid: str, langs: list[str]) -> tuple[str, int]:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(vid)
        try:
            transcript = transcript_list.find_transcript(langs)
        except Exception:
            transcript = next(iter(transcript_list))
        segments = transcript.fetch()
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Bu video için altyazı bulunamadı veya erişilemedi: {str(e)}",
        )

    lines = []
    last_sec = 0
    for s in segments:
        start = getattr(s, "start", s.get("start", 0) if isinstance(s, dict) else 0)
        text = getattr(s, "text", s.get("text", "") if isinstance(s, dict) else "")
        if text:
            sec = int(start)
            last_sec = max(last_sec, sec)
            lines.append(f"[{sec // 60:02d}:{sec % 60:02d}] {text.strip()}")

    return "\n".join(lines), max(1, round(last_sec / 60))


class AnalyzeReq(BaseModel):
    url: str
    lang: str = "Türkçe"
    caption_langs: list[str] = ["tr", "en"]


class ChatReq(BaseModel):
    transcript: str
    question: str


@app.post("/api/analyze")
@app.post("/analyze")
def analyze(req: AnalyzeReq):
    vid = extract_video_id(req.url)
    transcript_text, duration_min = fetch_transcript_text(vid, req.caption_langs)
    trimmed_text = transcript_text[:120_000]

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=trimmed_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.replace("{lang}", req.lang),
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        data = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI analiz hatası: {str(e)}")

    meta = data.setdefault("meta", {})
    meta["url"] = f"[https://youtu.be/](https://youtu.be/){vid}"
    meta["duration_min"] = duration_min
    data["_transcript"] = trimmed_text
    return data


@app.post("/api/chat")
@app.post("/chat")
def chat(req: ChatReq):
    system_msg = (
        "Yalnızca verilen video transkriptine dayanarak kısa ve net yanıtla. "
        "Her iddianın sonuna ilgili [mm:ss] zaman damgasını ekle. "
        "Konu transkriptte yoksa 'Bu videoda geçmiyor' de."
    )
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"TRANSKRİPT:\n{req.transcript[:120_000]}\n\nSORU: {req.question}",
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                temperature=0.2,
            ),
        )
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sohbet hatası: {str(e)}")