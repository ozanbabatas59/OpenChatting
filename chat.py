
import os
import logging
import threading
import inspect
from pathlib import Path

# =========================================================
# 1. ORTAM
# =========================================================

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
from llama_cpp import Llama

# =========================================================
# 2. AYARLAR
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "llm" / "Qwen3.5-4B-BF16.gguf"

APP_TITLE = "QWEN AI STUDIO"
CONTEXT_SIZE = 4096
GPU_LAYERS = 25
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = (
    "Sen profesyonel, güvenilir ve yardımcı bir Türkçe "
    "yapay zekâ asistanısın. Açık, doğru ve anlaşılır "
    "yanıtlar ver. Bilmediğin konularda bunu dürüstçe belirt."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("QwenStudio")

model_lock = threading.Lock()
llm = None
model_error = None


# =========================================================
# 3. GRADIO UYUMLULUK YARDIMCISI
# =========================================================

def uyumlu_ayarlar(fonksiyon, ayarlar):
    try:
        imza = inspect.signature(fonksiyon)
        parametreler = imza.parameters

        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parametreler.values()
        ):
            return ayarlar

        return {
            k: v for k, v in ayarlar.items()
            if k in parametreler
        }
    except (TypeError, ValueError):
        return ayarlar


def bilesen(bilesen_sinifi, **ayarlar):
    return bilesen_sinifi(
        **uyumlu_ayarlar(bilesen_sinifi, ayarlar)
    )


# =========================================================
# 4. MODEL
# =========================================================

def model_yukle():
    global llm, model_error

    if not MODEL_PATH.is_file():
        model_error = f"Model dosyası bulunamadı: {MODEL_PATH}"
        logger.error(model_error)
        return

    try:
        logger.info("Qwen modeli yükleniyor...")

        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=GPU_LAYERS,
            n_batch=512,
            verbose=False,
        )

        logger.info("Model başarıyla yüklendi.")

    except Exception as exc:
        model_error = str(exc)
        logger.exception("Model yükleme hatası")


model_yukle()


# =========================================================
# 5. GEÇMİŞ
# =========================================================

def gecmisi_duzenle(history):
    sonuc = []

    if not isinstance(history, list):
        return sonuc

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        sonuc.append({
            "role": role,
            "content": content,
        })

    sonuc = sonuc[-MAX_HISTORY_MESSAGES:]

    while sonuc and sonuc[0]["role"] != "user":
        sonuc.pop(0)

    return sonuc


# =========================================================
# 6. AKIŞKAN YANIT MOTORU
# =========================================================

def cevap_uret(
    mesaj,
    gecmis,
    sistem_mesaji,
    sicaklik,
    max_token,
):
    if not isinstance(mesaj, str) or not mesaj.strip():
        yield "Bir mesaj yazarak başlayabilirsin."
        return

    if llm is None:
        yield (
            "⚠️ **Model yüklenemedi.**\n\n"
            f"{model_error or 'Bilinmeyen hata.'}"
        )
        return

    if not isinstance(sistem_mesaji, str) or not sistem_mesaji.strip():
        sistem_mesaji = SYSTEM_PROMPT

    messages = [{
        "role": "system",
        "content": sistem_mesaji.strip(),
    }]

    messages.extend(gecmisi_duzenle(gecmis))

    messages.append({
        "role": "user",
        "content": mesaj.strip(),
    })

    answer = ""

    try:
        yield "✨ Düşünceler hazırlanıyor..."

        with model_lock:
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=int(max_token),
                temperature=float(sicaklik),
                top_p=0.9,
                stream=True,
            )

            for part in stream:
                choices = part.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if isinstance(content, str) and content:
                    answer += content
                    yield answer

        if not answer:
            yield "Model yanıt üretemedi. Tekrar deneyebilirsin."

    except Exception:
        logger.exception("Yanıt üretim hatası")

        if answer:
            yield answer + "\n\n⚠️ Üretim beklenmedik biçimde kesildi."
        else:
            yield (
                "⚠️ Yanıt oluşturulamadı. Model belleğini ve "
                "terminal hata kayıtlarını kontrol et."
            )


# =========================================================
# 7. MODERN NEON TASARIMI
# =========================================================

CSS = r"""
:root {
    --neon-blue: #60a5fa;
    --neon-cyan: #22d3ee;
    --neon-violet: #a78bfa;
    --panel-bg: rgba(15, 23, 42, 0.72);
    --panel-border: rgba(148, 163, 184, 0.17);
}

.gradio-container {
    max-width: 1250px !important;
    margin: 0 auto !important;
    padding: 18px !important;
    background: transparent !important;
}

body {
    background: #070b17 !important;
}

/* Arka plan ışıkları */
.gradio-container::before {
    content: "";
    position: fixed;
    inset: -30%;
    z-index: -2;
    pointer-events: none;
    background:
        radial-gradient(
            ellipse at 15% 15%,
            rgba(37, 99, 235, .18),
            transparent 35%
        ),
        radial-gradient(
            ellipse at 85% 30%,
            rgba(124, 58, 237, .16),
            transparent 32%
        ),
        radial-gradient(
            ellipse at 50% 90%,
            rgba(8, 145, 178, .12),
            transparent 36%
        );
    animation: ambientFlow 18s ease-in-out infinite alternate;
}

@keyframes ambientFlow {
    0% { transform: translate3d(-1%, -1%, 0) scale(1); }
    50% { transform: translate3d(2%, 1%, 0) scale(1.04); }
    100% { transform: translate3d(-1%, 2%, 0) scale(1.02); }
}

/* Üst başlık */
#hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 35px 20px 27px;
    margin-bottom: 22px;
    border: 1px solid var(--panel-border);
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        rgba(15, 23, 42, .9),
        rgba(30, 41, 75, .48),
        rgba(15, 23, 42, .82)
    );
    box-shadow: 0 18px 65px rgba(0, 0, 0, .2);
}

#hero::before {
    content: "";
    position: absolute;
    width: 70%;
    height: 1px;
    left: 15%;
    top: 0;
    background: linear-gradient(
        90deg,
        transparent,
        #60a5fa,
        #a78bfa,
        transparent
    );
    box-shadow: 0 0 24px #60a5fa;
    animation: lineGlow 4s ease-in-out infinite alternate;
}

@keyframes lineGlow {
    from { opacity: .35; transform: scaleX(.65); }
    to { opacity: 1; transform: scaleX(1); }
}

#hero .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 7px 13px;
    border: 1px solid rgba(96, 165, 250, .25);
    border-radius: 999px;
    background: rgba(37, 99, 235, .1);
    color: #bfdbfe;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
}

#hero .live-dot {
    width: 7px;
    height: 7px;
    display: inline-block;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 12px #34d399;
    animation: livePulse 1.8s infinite;
}

@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .45; transform: scale(.75); }
}

#hero h1 {
    margin: 22px 0 9px;
    font-size: clamp(34px, 5vw, 58px);
    font-weight: 850;
    letter-spacing: -2.4px;
    line-height: 1.08;
    color: #f8fafc;
}

#hero h1 span {
    background: linear-gradient(
        90deg,
        #93c5fd,
        #67e8f9,
        #c4b5fd,
        #93c5fd
    );
    background-size: 250% auto;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 7s linear infinite;
}

@keyframes gradientText {
    to { background-position: 250% center; }
}

#hero .subtitle {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.8;
}

#hero .hero-footer {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 21px;
}

#hero .hero-chip {
    padding: 8px 12px;
    border: 1px solid rgba(148, 163, 184, .15);
    border-radius: 10px;
    background: rgba(15, 23, 42, .6);
    color: #cbd5e1;
    font-size: 12px;
}

/* Bilgi kartları */
#privacy-panel {
    position: relative;
    padding: 17px 20px;
    border: 1px solid rgba(34, 211, 238, .16);
    border-radius: 17px;
    background: linear-gradient(
        120deg,
        rgba(8, 47, 73, .48),
        rgba(15, 23, 42, .78)
    );
    margin-bottom: 15px;
    transition: border-color .3s, transform .3s;
}

#privacy-panel:hover {
    border-color: rgba(34, 211, 238, .4);
    transform: translateY(-2px);
}

#privacy-panel strong {
    color: #a5f3fc;
    font-size: 14px;
}

#privacy-panel p {
    color: #a5b4c8;
    font-size: 12px;
    line-height: 1.8;
    margin: 8px 0 0;
}

/* Durum kartı */
#model-status {
    padding: 12px 15px;
    border: 1px solid var(--panel-border);
    border-radius: 13px;
    background: rgba(15, 23, 42, .55);
    color: #cbd5e1;
    font-size: 12px;
}

/* Ayarlar paneli */
#settings-panel {
    border: 1px solid var(--panel-border) !important;
    border-radius: 17px !important;
    background: var(--panel-bg) !important;
    overflow: hidden;
    transition: border-color .25s;
}

#settings-panel:hover {
    border-color: rgba(96, 165, 250, .35) !important;
}

/* Sohbet alanı */
#chat-panel {
    margin-top: 14px;
    border: 1px solid var(--panel-border);
    border-radius: 21px;
    padding: 10px;
    background: rgba(10, 15, 30, .55);
    box-shadow: 0 15px 55px rgba(0, 0, 0, .13);
}

#chat-panel .message {
    border-radius: 17px !important;
    line-height: 1.75 !important;
}

#chat-panel textarea {
    border-radius: 14px !important;
}

/* Düğmeler */
.gradio-container button {
    transition:
        transform .2s ease,
        box-shadow .2s ease,
        border-color .2s ease !important;
}

.gradio-container button:hover {
    transform: translateY(-1px);
}

.gradio-container button:active {
    transform: scale(.98);
}

/* Kaydırma çubukları */
* {
    scrollbar-width: thin;
    scrollbar-color: #334155 transparent;
}

/* Alt bilgi */
#footer {
    padding: 22px 10px;
    text-align: center;
    color: #64748b;
    font-size: 11px;
    letter-spacing: .5px;
}

@media (max-width: 640px) {
    .gradio-container {
        padding: 9px !important;
    }

    #hero {
        padding: 27px 12px 22px;
        border-radius: 19px;
    }

    #hero h1 {
        letter-spacing: -1.5px;
    }

    #privacy-panel {
        padding: 14px;
    }

    #chat-panel {
        padding: 5px;
    }
}

/* Hareket hassasiyeti olan kullanıcılar */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}
"""

tema = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="lg",
)

# =========================================================
# 8. ARAYÜZ
# =========================================================

with gr.Blocks(title=APP_TITLE) as demo:

    gr.HTML("""
        <section id="hero">
            <div class="eyebrow">
                <span class="live-dot"></span>
                LOCAL INTELLIGENCE SYSTEM
            </div>

            <h1>QWEN <span>AI STUDIO</span></h1>

            <div class="subtitle">
                Yeni nesil yerel yapay zekâ deneyimi.<br>
                Güçlü. Akıcı. Kontrol tamamen sende.
            </div>

            <div class="hero-footer">
                <div class="hero-chip">✦ Yerel model</div>
                <div class="hero-chip">ϟ Akışkan yanıtlar</div>
                <div class="hero-chip">◈ Gizlilik odaklı</div>
            </div>
        </section>
    """)

    gr.HTML("""
        <div id="privacy-panel">
            <strong>◈ &nbsp; YEREL ÇALIŞMA MODU</strong>
            <p>
                Model yerel GGUF dosyasından çalışır ve bu arayüz
                yanıt üretmek için harici bir AI API'si kullanmaz.
                Bu gösterge, bilgisayarın tüm ağ trafiğinin kapalı
                olduğunu veya verilerin hiçbir yere kaydedilmediğini
                garanti etmez. Gerçek izolasyon için güvenlik duvarı
                kurallarını ayrıca yapılandır.
            </p>
        </div>
    """)

    if llm is not None:
        status = (
            f"● MODEL ONLINE  /  CONTEXT {CONTEXT_SIZE}  /  "
            f"GPU LAYERS {GPU_LAYERS}"
        )
    else:
        status = "● MODEL OFFLINE / MODEL DOSYASINI KONTROL ET"

    gr.Markdown(status, elem_id="model-status")

    with gr.Accordion(
        "⚙️  MODEL KONTROL MERKEZİ",
        open=False,
        elem_id="settings-panel",
    ):
        sistem_mesaji = gr.Textbox(
            value=SYSTEM_PROMPT,
            label="Sistem talimatı",
            lines=4,
        )

        with gr.Row():
            sicaklik = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.6,
                step=0.1,
                label="✧ Yaratıcılık",
            )

            max_token = gr.Slider(
                minimum=128,
                maximum=2048,
                value=1024,
                step=128,
                label="ϟ Maksimum çıktı token sayısı",
            )

    with gr.Group(elem_id="chat-panel"):

        chatbot = bilesen(
            gr.Chatbot,
            height=570,
            placeholder=(
                "## ✦ Hoş geldin\n\n"
                "Yerel yapay zekân hazır. Bir soru sorarak "
                "sohbeti başlatabilirsin."
            ),
        )

        textbox = bilesen(
            gr.Textbox,
            placeholder="Bir şey sor...  ↵",
            container=False,
            autofocus=True,
            submit_btn="Gönder",
            stop_btn="Durdur",
        )

        chat_ayar = {
            "fn": cevap_uret,
            "type": "messages",
            "chatbot": chatbot,
            "textbox": textbox,
            "additional_inputs": [
                sistem_mesaji,
                sicaklik,
                max_token,
            ],
            "retry_btn": "Yeniden dene",
            "undo_btn": "Geri al",
            "clear_btn": "Sohbeti temizle",
            "concurrency_limit": 1,
        }

        chat_ayar = uyumlu_ayarlar(
            gr.ChatInterface,
            chat_ayar,
        )

        gr.ChatInterface(**chat_ayar)

    gr.HTML("""
        <div id="footer">
            QWEN AI STUDIO &nbsp;·&nbsp; LOCAL INFERENCE
            <br><br>
            YOUR MACHINE. YOUR MODEL. YOUR CONTROL.
        </div>
    """)


# =========================================================
# 9. BAŞLAT
# =========================================================

if __name__ == "__main__":

    demo.queue(default_concurrency_limit=1)

    logger.info("Gradio sürümü: %s", gr.__version__)
    logger.info("Qwen AI Studio başlatılıyor...")

    launch_ayar = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "inbrowser": True,
        "share": False,
        "theme": tema,
        "css": CSS,
        "show_error": True,
    }

    launch_ayar = uyumlu_ayarlar(
        demo.launch,
        launch_ayar,
    )

    demo.launch(**launch_ayar)
