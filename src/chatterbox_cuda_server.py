import json
import math
import os
import re
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "tts" / "BricksDisplay-chatterbox-multilingual-ONNX-q4"
ONNX_DIR = MODEL_DIR / "onnx"
HOST = "0.0.0.0"
PORT = 8081
SAMPLE_RATE = 24000
PROGRESS_FILE = ROOT / "runtime" / "tts_load_status.json"
CUDA_MEM_LIMIT_MB = int(os.environ.get("CHATTERBOX_CUDA_MEM_LIMIT_MB", "2300"))
MAX_GENERATION_TOKENS = int(os.environ.get("CHATTERBOX_MAX_GENERATION_TOKENS", "220"))
SILENCE_TOKEN = 4299
START_SPEECH_TOKEN = 6561
EOS_TOKENS = {2, 6562}

tokenizer = None
cangjie = None
sessions = {}
speaker_cache = {}
load_error = None
load_started = time.time()


def write_progress(percent, step, status="loading", error=""):
    payload = {
        "status": status,
        "percent": int(percent),
        "step": step,
        "updated_at": time.time(),
        "uptime_sec": round(time.time() - load_started, 1),
        "error": str(error or ""),
    }
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def read_wav_float32(path):
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"Reference audio must be {SAMPLE_RATE}Hz, got {rate}")
    if width != 2:
        raise RuntimeError(f"Expected 16-bit PCM wav, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.reshape(1, -1).astype(np.float32)


def wav_bytes(samples):
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * np.where(data < 0, 32768.0, 32767.0)).astype("<i2")
    size = pcm.nbytes
    header = bytearray(44)
    header[0:4] = b"RIFF"
    header[4:8] = (36 + size).to_bytes(4, "little")
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    header[16:20] = (16).to_bytes(4, "little")
    header[20:22] = (1).to_bytes(2, "little")
    header[22:24] = (1).to_bytes(2, "little")
    header[24:28] = SAMPLE_RATE.to_bytes(4, "little")
    header[28:32] = (SAMPLE_RATE * 2).to_bytes(4, "little")
    header[32:34] = (2).to_bytes(2, "little")
    header[34:36] = (16).to_bytes(2, "little")
    header[36:40] = b"data"
    header[40:44] = size.to_bytes(4, "little")
    return bytes(header) + pcm.tobytes()


def load_cangjie():
    table = {}
    reverse = {}
    entries = json.loads((MODEL_DIR / "Cangjie5_TC.json").read_text(encoding="utf-8"))
    for entry in entries:
        parts = entry.split("\t")
        if len(parts) != 2:
            continue
        word, code = parts
        table[word] = code
        reverse.setdefault(code, []).append(word)
    return table, reverse


def sanitize_text(text):
    text = re.sub(r"【[^】]*】", "", str(text or ""))
    text = re.sub(r"\[[a-z]{2}\]", "", text, flags=re.I)
    text = re.sub(r"[，、。！？；：“”‘’《》（）【】…—~·]", " ", text)
    text = re.sub(r"""[,.!?;:"'()[\]{}<>|\\/_+=*@#$%^&`-]""", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_chinese(text):
    table, reverse = cangjie
    out = []
    for ch in text:
        if "\u3400" <= ch <= "\u9fff" and ch in table:
            code = table[ch]
            variants = reverse.get(code, [])
            idx = variants.index(ch) if ch in variants else 0
            indexed = code + (str(idx) if idx > 0 else "")
            out.append("".join(f"[cj_{c}]" for c in indexed) + "[cj_.]")
        else:
            out.append(ch)
    return "".join(out)


def detect_language(text, requested):
    if requested and requested != "z":
        return requested
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    return "en"


def prepare_text(text, lang):
    clean = sanitize_text(text)
    if not clean:
        return "[zh][cj_o][cj_k][cj_.]"
    language = detect_language(clean, lang)
    encoded = encode_chinese(clean) if language == "zh" else clean
    return f"[{language}]{encoded}"


def session_options():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def providers():
    return [("CUDAExecutionProvider", {
        "device_id": 0,
        "gpu_mem_limit": CUDA_MEM_LIMIT_MB * 1024 * 1024,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
    }), "CPUExecutionProvider"]


def load_model():
    global tokenizer, cangjie, load_error
    if sessions:
        return
    try:
        write_progress(5, "初始化 ONNX Runtime")
        try:
            ort.preload_dlls(directory="")
        except Exception:
            pass
        write_progress(12, "加载 tokenizer")
        tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        write_progress(18, "加载中文编码表")
        cangjie = load_cangjie()
        model_steps = (
            ("speech_encoder", 35),
            ("embed_tokens", 50),
            ("language_model", 72),
            ("conditional_decoder", 90),
        )
        for name, percent in model_steps:
            write_progress(percent - 10, f"加载 {name}.onnx")
            sessions[name] = ort.InferenceSession(
                str(ONNX_DIR / f"{name}.onnx"),
                sess_options=session_options(),
                providers=providers(),
            )
            write_progress(percent, f"{name}.onnx 已加载")
        write_progress(95, "缓存默认声线")
        get_speaker("female")
        write_progress(100, "语音模型已就绪", status="ok")
    except Exception as exc:
        load_error = exc
        write_progress(100, "语音模型加载失败", status="error", error=exc)
        raise


def voice_path(voice):
    key = str(voice or "female").lower()
    if key in {"female", "zf_001", "zh_female"}:
        return MODEL_DIR / "female_voice.wav"
    return MODEL_DIR / "default_voice.wav"


def get_speaker(voice):
    load_model() if not sessions else None
    path = voice_path(voice)
    key = path.name
    if key not in speaker_cache:
        audio = read_wav_float32(path)
        out = sessions["speech_encoder"].run(None, {"audio_values": audio})
        names = [x.name for x in sessions["speech_encoder"].get_outputs()]
        speaker_cache[key] = dict(zip(names, out))
    return speaker_cache[key]


def position_ids_for_prefill(ids):
    pos = []
    cur = 0
    for token in ids:
        if token >= START_SPEECH_TOKEN:
            pos.append(0)
        else:
            pos.append(cur)
            cur += 1
    return np.asarray([pos], dtype=np.int64)


def position_id_for_next(sequence):
    last_start = max(i for i, token in enumerate(sequence) if token == START_SPEECH_TOKEN)
    return np.asarray([[len(sequence) - last_start - 1]], dtype=np.int64)


def empty_past():
    past = {}
    for i in range(30):
        past[f"past_key_values.{i}.key"] = np.zeros((1, 16, 0, 64), dtype=np.float32)
        past[f"past_key_values.{i}.value"] = np.zeros((1, 16, 0, 64), dtype=np.float32)
    return past


def outputs_to_past(outputs):
    names = [x.name for x in sessions["language_model"].get_outputs()]
    mapped = dict(zip(names, outputs))
    past = {}
    for i in range(30):
        past[f"past_key_values.{i}.key"] = mapped[f"present.{i}.key"]
        past[f"past_key_values.{i}.value"] = mapped[f"present.{i}.value"]
    return mapped["logits"], past


def apply_repetition_penalty(logits, tokens, penalty):
    if penalty <= 1.0:
        return logits
    for token in set(tokens):
        if 0 <= token < logits.shape[-1]:
            logits[token] = logits[token] * penalty if logits[token] < 0 else logits[token] / penalty
    return logits


def sample_token(logits, temperature=0.8, top_p=0.95, repetition_tokens=None, repetition_penalty=1.2):
    logits = logits.astype(np.float64)
    if repetition_tokens:
        logits = apply_repetition_penalty(logits, repetition_tokens, repetition_penalty)
    logits = logits / max(float(temperature), 1e-5)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    keep = np.cumsum(sorted_probs) <= float(top_p)
    keep[0] = True
    kept = order[keep]
    kept_probs = probs[kept]
    kept_probs /= np.sum(kept_probs)
    return int(np.random.choice(kept, p=kept_probs))


def synthesize(text, options):
    load_model()
    lang = options.get("lang") or "zh"
    voice = options.get("voice") or "female"
    prepared = prepare_text(text, lang)
    enc = tokenizer.encode(prepared)
    ids = enc.ids
    input_ids = np.asarray([ids], dtype=np.int64)
    pos = position_ids_for_prefill(ids)
    exaggeration = np.asarray([float(options.get("exaggeration") or 0.5)], dtype=np.float32)
    speaker = get_speaker(voice)

    embeds = sessions["embed_tokens"].run(None, {
        "input_ids": input_ids,
        "position_ids": pos,
        "exaggeration": exaggeration,
    })[0]
    embeds = np.concatenate([speaker["audio_features"], embeds], axis=1)
    past = empty_past()
    attn = np.ones((1, embeds.shape[1]), dtype=np.int64)
    outputs = sessions["language_model"].run(None, {"inputs_embeds": embeds, "attention_mask": attn, **past})
    logits, past = outputs_to_past(outputs)

    sequence = list(ids)
    generated = []
    max_tokens = int(options.get("max_new_tokens") or 0)
    if max_tokens <= 0:
        max_tokens = max(80, min(420, math.ceil(len(prepared) * 3.2)))
    max_tokens = min(max_tokens, MAX_GENERATION_TOKENS)

    for _ in range(max_tokens):
        next_id = sample_token(
            logits[0, -1],
            temperature=float(options.get("temperature") or 0.8),
            top_p=float(options.get("top_p") or 0.95),
            repetition_tokens=generated,
            repetition_penalty=float(options.get("repetition_penalty") or 1.2),
        )
        sequence.append(next_id)
        generated.append(next_id)
        if next_id in EOS_TOKENS:
            break

        step_ids = np.asarray([[next_id]], dtype=np.int64)
        step_pos = position_id_for_next(sequence)
        step_embeds = sessions["embed_tokens"].run(None, {
            "input_ids": step_ids,
            "position_ids": step_pos,
            "exaggeration": exaggeration,
        })[0]
        past_len = next(iter(past.values())).shape[2]
        attn = np.ones((1, past_len + 1), dtype=np.int64)
        outputs = sessions["language_model"].run(None, {"inputs_embeds": step_embeds, "attention_mask": attn, **past})
        logits, past = outputs_to_past(outputs)

    speech_new = generated[:-1] if generated and generated[-1] in EOS_TOKENS else generated
    speech_tokens = np.concatenate([
        speaker["audio_tokens"].astype(np.int64),
        np.asarray([speech_new], dtype=np.int64),
        np.full((1, 3), SILENCE_TOKEN, dtype=np.int64),
    ], axis=1)
    waveform = sessions["conditional_decoder"].run(None, {
        "speech_tokens": speech_tokens,
        "speaker_embeddings": speaker["speaker_embeddings"],
        "speaker_features": speaker["speaker_features"],
    })[0]
    return wav_bytes(waveform)


def send_json(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_wav(handler, body):
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/wav")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            send_json(self, 200, {
                "status": "error" if load_error else ("ok" if sessions else "loading"),
                "model": "BricksDisplay/chatterbox-multilingual-ONNX-q4",
                "backend": "onnxruntime-gpu",
                "device": "cuda",
                "providers": ort.get_available_providers(),
                "session_providers": sessions["language_model"].get_providers() if sessions else [],
                "cuda_mem_limit_mb": CUDA_MEM_LIMIT_MB,
                "max_generation_tokens": MAX_GENERATION_TOKENS,
                "uptime_sec": round(time.time() - load_started, 1),
                "error": str(load_error) if load_error else "",
            })
            return
        if parsed.path == "/tts":
            params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            audio = synthesize(params.get("text", ""), params)
            send_wav(self, audio)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/tts":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            audio = synthesize(data.get("text", ""), data)
            send_wav(self, audio)
        except Exception as exc:
            send_json(self, 500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    try:
        load_model()
    except Exception as exc:
        print(f"[cuda-tts] load failed: {exc}", flush=True)
    print(f"[cuda-tts] listening on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
