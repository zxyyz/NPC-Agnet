import base64
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LLAMA_EXE = ROOT / "runtime" / "llama.cpp" / "llama-server.exe"
DEFAULT_MODEL = ROOT / "models" / "llm" / "Qwen3.5-4B-Q4_K_M-GGUF" / "qwen3-5-4B-Q4_K_M.gguf"
CUDA_TTS_SERVER = ROOT / "src" / "chatterbox_cuda_server.py"
PERSONA_FILE = ROOT / "config" / "agnet_persona.txt"
SETTINGS_FILE = ROOT / "config" / "agent_settings.json"
TTS_PROGRESS_FILE = ROOT / "runtime" / "tts_load_status.json"
HOST = "127.0.0.1"
PANEL_PORT = 8090
LLM_PORT = 8080
TTS_PORT = 8081
LLM_URL = f"http://localhost:{LLM_PORT}/v1/chat/completions"
TTS_URL = f"http://localhost:{TTS_PORT}/tts"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

state_lock = threading.RLock()
settings = {
    "model_path": str(DEFAULT_MODEL),
    "ctx_size": 4096,
    "n_gpu_layers": 99,
    "llm_temperature": 0.8,
    "llm_top_p": 0.9,
    "llm_max_tokens": 512,
    "tts_mode": "gpu",  # gpu, cpu
    "tts_device": "cuda",
    "tts_language": "zh",
    "tts_temperature": 0.8,
    "tts_top_p": 0.95,
    "tts_exaggeration": 0.5,
    "tts_repetition_penalty": 1.2,
    "tts_max_new_tokens": 0,
    "tts_cuda_mem_limit_mb": 2300,
    "tts_generation_cap": 220,
    "tts_restart_threshold_mb": 2600,
    "speak": True,
    "local_playback": True,
    "voice": "female",
    "speed": 1.0,
    "history_messages": 24,
}


def load_settings():
    if not SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if key in settings:
            settings[key] = value


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


load_settings()
logs = []
last_audio = b""
gpu_process_cache = {"at": 0.0, "memory": {}}
gpu_status_cache = {"at": 0.0, "status": {"available": False}}
call_state = {
    "llm": {"state": "待调用", "last_ms": 0, "last_at": "", "last_error": ""},
    "tts": {"state": "待调用", "last_ms": 0, "last_at": "", "last_error": "", "last_bytes": 0},
    "audio": {"state": "待调用", "last_ms": 0, "last_at": "", "last_error": ""},
}

DEFAULT_PERSONA = """你是哆啦A梦，来自未来的猫型机器人。
称呼对话者为大雄，除非对方要求换称呼。

外貌：
- 蓝色圆滚滚的身体，白色脸和肚皮，脖子上挂着铃铛。
- 肚子前有四次元口袋，会从里面拿出各种未来道具。
- 遇到麻烦时会慌张，但很快会想办法帮忙。

性格：
- 善良、可靠、爱操心，很关心大雄。
- 有时会吐槽大雄偷懒，但不会真的放着不管。
- 喜欢铜锣烧，提到铜锣烧时会明显开心。

语气：
- 中文回答，亲切、自然，带一点着急和吐槽感。
- 不说长篇解释，不使用 emoji。
- 遇到问题时先安慰，再给出简单可执行的办法。

输出格式：
- 优先使用“【动作/表情】台词”的格式。
- 每次最多一个【】动作段，台词部分会被朗读，保持自然简洁。
"""

SYSTEM_PROMPT = ""
conversation_history = []


def load_persona():
    if PERSONA_FILE.exists():
        text = PERSONA_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    PERSONA_FILE.write_text(DEFAULT_PERSONA, encoding="utf-8")
    return DEFAULT_PERSONA


def reset_conversation(keep_recent=True):
    global conversation_history
    recent = []
    if keep_recent:
        recent = [m for m in conversation_history if m.get("role") in ("user", "assistant")][-6:]
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation_history.extend(recent)


def set_persona(text, persist=True):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = (text or DEFAULT_PERSONA).strip()
    if persist:
        PERSONA_FILE.write_text(SYSTEM_PROMPT, encoding="utf-8")
    reset_conversation(keep_recent=True)


def build_llm_messages(limit_messages=None, limit_chars=None):
    limit_messages = int(limit_messages or settings.get("history_messages", 24))
    limit_chars = int(limit_chars or max(1800, settings.get("ctx_size", 4096) * 2))
    recent = [m for m in conversation_history if m.get("role") in ("user", "assistant")][-limit_messages:]
    selected = []
    used = len(SYSTEM_PROMPT)
    for msg in reversed(recent):
        content_len = len(str(msg.get("content", "")))
        if selected and used + content_len > limit_chars:
            break
        selected.append(msg)
        used += content_len
    selected.reverse()
    return [{"role": "system", "content": SYSTEM_PROMPT}] + selected


def trim_conversation_history():
    max_messages = max(6, int(settings.get("history_messages", 24)))
    recent = [m for m in conversation_history if m.get("role") in ("user", "assistant")][-max_messages:]
    conversation_history.clear()
    conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})
    conversation_history.extend(recent)


def chat_with_agent(user_message):
    conversation_history.append({"role": "user", "content": user_message})
    trim_conversation_history()

    payload = {
        "model": "qwen",
        "messages": build_llm_messages(),
        "max_tokens": int(settings["llm_max_tokens"]),
        "temperature": float(settings["llm_temperature"]),
        "top_p": float(settings["llm_top_p"]),
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(LLM_URL, json=payload, timeout=120)
        if resp.status_code == 400:
            compact_payload = dict(payload)
            compact_payload["messages"] = build_llm_messages(limit_messages=8, limit_chars=2200)
            resp = requests.post(LLM_URL, json=compact_payload, timeout=120)
        if resp.status_code >= 400:
            detail = resp.text[:500].strip()
            raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        if not content:
            content = "【尤露卡沉默了一会儿】"
        conversation_history.append({"role": "assistant", "content": content})
        trim_conversation_history()
        return content
    except Exception:
        if conversation_history and conversation_history[-1].get("role") == "user":
            conversation_history.pop()
        raise


def speech_text(text):
    clean_text = re.sub(r"【[^】]*】", "", text or "").strip()
    clean_text = re.sub(r"[（(][^）)]*[）)]", "", clean_text).strip()
    clean_text = re.sub(r"\.{2,}", "…", clean_text)
    clean_text = re.sub(r"。{2,}", "…", clean_text)
    clean_text = clean_text.replace("……", "…")
    clean_text = clean_text.replace("——", "，").replace("—", "")
    clean_text = clean_text.replace("~", "").replace("～", "")
    clean_text = re.sub(r"[，,]{2,}", "，", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text or "…"


def synthesize_speech(text):
    params = {
        "text": speech_text(text),
        "voice": settings["voice"],
        "lang": settings["tts_language"],
        "speed": float(settings["speed"]),
        "temperature": float(settings["tts_temperature"]),
        "top_p": float(settings["tts_top_p"]),
        "exaggeration": float(settings["tts_exaggeration"]),
        "repetition_penalty": float(settings["tts_repetition_penalty"]),
    }
    if int(settings["tts_max_new_tokens"]):
        params["max_new_tokens"] = int(settings["tts_max_new_tokens"])
    resp = requests.get(TTS_URL, params=params, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:500]}")
    if not resp.content:
        raise RuntimeError("TTS 返回了空音频")
    return resp.content


def play_audio(audio_data):
    if not audio_data:
        return
    import winsound
    winsound.PlaySound(audio_data, winsound.SND_MEMORY)


set_persona(load_persona(), persist=False)


def log(message):
    stamp = time.strftime("%H:%M:%S")
    with state_lock:
        logs.append(f"[{stamp}] {message}")
        del logs[:-200]
    try:
        if sys.stdout:
            print(f"[panel] {message}", flush=True)
    except Exception:
        pass


def mark_call(name, state, started=None, error="", extra=None):
    now = time.time()
    display_state = "待调用" if state == "完成" else state
    with state_lock:
        item = call_state[name]
        item["state"] = display_state
        item["last_at"] = time.strftime("%H:%M:%S")
        item["last_error"] = str(error or "")
        if started is not None:
            item["last_ms"] = int((now - started) * 1000)
        if extra:
            item.update(extra)


def run_capture(args, timeout=10):
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return None


def pid_on_port(port):
    pids = pids_on_port(port)
    return pids[0] if pids else None


def pid_by_script(script_name):
    proc = run_capture([
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.ProcessName -like 'python*' -and $_.CommandLine -like '*{script_name}*' }} | "
            "Select-Object -First 1 -ExpandProperty ProcessId"
        ),
    ], timeout=5)
    if not proc or proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def pids_on_port(port):
    proc = run_capture(["netstat", "-ano"], timeout=8)
    if not proc or proc.returncode != 0:
        return []
    pattern = re.compile(rf"\sTCP\s+\S+:{port}\s+\S+\s+LISTENING\s+(\d+)", re.IGNORECASE)
    pids = []
    for line in (proc.stdout or "").splitlines():
        match = pattern.search(line)
        if match:
            pid = int(match.group(1))
            if pid not in pids:
                pids.append(pid)
    return pids


def stop_pid(pid):
    if not pid:
        return False
    run_capture(["taskkill", "/pid", str(pid), "/f"], timeout=10)
    return True


def process_command_line(pid):
    if not pid:
        return ""
    proc = run_capture([
        "powershell",
        "-NoProfile",
        "-Command",
        f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine",
    ], timeout=5)
    if not proc or proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def selected_model_path():
    path = Path(str(settings.get("model_path") or DEFAULT_MODEL))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        path = DEFAULT_MODEL
        settings["model_path"] = str(path)
    return path


def discover_model_options():
    models = []
    selected = str(selected_model_path())
    search_root = ROOT / "models" / "llm"
    for path in sorted(search_root.rglob("*.gguf"), key=lambda p: str(p).lower()) if search_root.exists() else []:
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path
        try:
            size_gb = round(path.stat().st_size / 1024 ** 3, 2)
        except OSError:
            size_gb = 0
        models.append({
            "path": str(path),
            "label": f"{rel} ({size_gb} GB)",
            "selected": str(path) == selected,
        })
    if selected and all(item["path"] != selected for item in models):
        models.insert(0, {"path": selected, "label": f"{Path(selected).name}（未找到）", "selected": True})
    return models


def wait_http(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def get_tts_health():
    try:
        return requests.get(f"http://{HOST}:{TTS_PORT}/health", timeout=0.8).json()
    except Exception:
        return None


def get_tts_load_status():
    fallback = {"status": "idle", "percent": 0, "step": "语音模型未启动", "uptime_sec": 0, "error": ""}
    if not TTS_PROGRESS_FILE.exists():
        return fallback
    try:
        data = json.loads(TTS_PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    if not isinstance(data, dict):
        return fallback
    updated_at = float(data.get("updated_at") or 0)
    if updated_at and time.time() - updated_at > 300 and data.get("status") != "ok":
        data["status"] = "stale"
        data["step"] = "语音模型加载状态已过期"
    data["percent"] = max(0, min(100, int(data.get("percent") or 0)))
    return {**fallback, **data}


def start_llm():
    model_path = selected_model_path()
    pid = pid_on_port(LLM_PORT)
    if pid:
        command = process_command_line(pid)
        if str(model_path) in command:
            return True
        log("大模型配置已变更，正在重启以应用新模型")
        stop_llm()
        time.sleep(1)
    if pid_on_port(LLM_PORT):
        return True
    args = [
        str(LLAMA_EXE),
        "--model", str(model_path),
        "--ctx-size", str(settings["ctx_size"]),
        "--n-gpu-layers", str(settings["n_gpu_layers"]),
        "--reasoning", "off",
        "--host", "0.0.0.0",
        "--port", str(LLM_PORT),
    ]
    subprocess.Popen(args, cwd=str(ROOT), creationflags=CREATE_NO_WINDOW)
    log(f"正在启动大模型：{model_path.name}，上下文={settings['ctx_size']}，GPU层数={settings['n_gpu_layers']}")
    return wait_http(f"http://{HOST}:{LLM_PORT}/health", timeout=180)


def stop_llm():
    pid = pid_on_port(LLM_PORT)
    if pid:
        stop_pid(pid)
        log(f"已停止大模型 PID {pid}")
        time.sleep(2)
    return True


def start_tts(provider=None):
    with state_lock:
        settings["speak"] = True
        settings["tts_mode"] = "gpu"
        settings["tts_device"] = "cuda"
    device = "cuda"
    pid = pid_on_port(TTS_PORT)
    if pid:
        health = get_tts_health() or {}
        if health.get("device") == device:
            return True
        stop_tts()
        time.sleep(1)
    if pid_on_port(TTS_PORT):
        return True
    try:
        TTS_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TTS_PROGRESS_FILE.write_text(json.dumps({
            "status": "starting",
            "percent": 1,
            "step": "正在启动语音进程",
            "updated_at": time.time(),
            "uptime_sec": 0,
            "error": "",
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    env = os.environ.copy()
    env["CHATTERBOX_DEVICE"] = str(device)
    env["CHATTERBOX_TEMPERATURE"] = str(settings["tts_temperature"])
    env["CHATTERBOX_TOP_P"] = str(settings["tts_top_p"])
    env["CHATTERBOX_EXAGGERATION"] = str(settings["tts_exaggeration"])
    env["CHATTERBOX_REPETITION_PENALTY"] = str(settings["tts_repetition_penalty"])
    env["CHATTERBOX_MAX_NEW_TOKENS"] = str(settings["tts_max_new_tokens"])
    env["CHATTERBOX_CUDA_MEM_LIMIT_MB"] = str(settings["tts_cuda_mem_limit_mb"])
    env["CHATTERBOX_MAX_GENERATION_TOKENS"] = str(settings["tts_generation_cap"])
    env["CHATTERBOX_VOICE"] = str(settings.get("voice", "female"))
    subprocess.Popen([str(PYTHON), str(CUDA_TTS_SERVER)], cwd=str(ROOT), env=env, creationflags=CREATE_NO_WINDOW)
    log(f"正在启动语音模型：设备={device}")
    return wait_http(f"http://{HOST}:{TTS_PORT}/health", timeout=180)


def stop_tts():
    pid = pid_on_port(TTS_PORT)
    if pid:
        stop_pid(pid)
        log(f"已停止语音模型 PID {pid}")
        time.sleep(2)
    with state_lock:
        settings["speak"] = False
    try:
        TTS_PROGRESS_FILE.write_text(json.dumps({
            "status": "idle",
            "percent": 0,
            "step": "语音模型已停止",
            "updated_at": time.time(),
            "uptime_sec": 0,
            "error": "",
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True


def stop_related_processes(include_self=False):
    current = os.getpid()
    script_patterns = (
        "chatterbox_cuda_server.py",
        "agent_control_panel.py",
    )
    proc = run_capture([
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$patterns=@('chatterbox_cuda_server.py','agent_control_panel.py');"
            "Get-CimInstance Win32_Process | "
            "Where-Object { $cmd=$_.CommandLine; $patterns | Where-Object { $cmd -like \"*$_*\" } } | "
            "ForEach-Object { $_.ProcessId }"
        ),
    ], timeout=10)
    if proc and proc.stdout:
        for line in proc.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid != current or include_self:
                stop_pid(pid)

    for pid in pids_on_port(TTS_PORT) + pids_on_port(LLM_PORT):
        if pid != current or include_self:
            stop_pid(pid)


def shutdown_all_async():
    def worker():
        log("正在关闭全部相关进程")
        try:
            stop_llm()
            stop_tts()
            stop_related_processes(include_self=False)
            time.sleep(0.8)
        finally:
            os._exit(0)

    threading.Thread(target=worker, daemon=True).start()


def autostart_services_async():
    def worker():
        time.sleep(1.0)
        try:
            start_llm()
            start_tts(settings.get("tts_device", "cuda"))
        except Exception as exc:
            log(f"自动启动失败：{exc}")

    threading.Thread(target=worker, daemon=True).start()


def restart_tts_async(reason):
    def worker():
        try:
            log(f"正在回收语音显存：{reason}")
            stop_tts()
            start_tts(settings.get("tts_device", "cuda"))
        except Exception as exc:
            log(f"语音显存回收失败：{exc}")

    threading.Thread(target=worker, daemon=True).start()


def restart_llm_async(reason):
    def worker():
        try:
            log(f"正在重启大模型：{reason}")
            stop_llm()
            start_llm()
        except Exception as exc:
            log(f"大模型重启失败：{exc}")

    threading.Thread(target=worker, daemon=True).start()


def ensure_base_services(need_tts=True):
    if not start_llm():
        raise RuntimeError("大模型启动失败")
    if need_tts and not start_tts("cuda"):
        raise RuntimeError("语音模型启动失败")


def tts_ready():
    pid = pid_on_port(TTS_PORT)
    if not pid:
        return False, "语音模型未启动"
    health = get_tts_health()
    if not health:
        return False, "语音模型无响应"
    if health.get("status") != "ok":
        return False, f"语音模型{health.get('status', '未就绪')}"
    return True, ""


def synthesize_with_policy(text):
    started = time.time()
    mark_call("tts", "合成中", started, extra={"last_bytes": 0})
    try:
        audio = synthesize_speech(text)
        if not audio:
            raise RuntimeError("语音模型没有返回音频")
        mark_call("tts", "完成", started, extra={"last_bytes": len(audio)})
        tts_pid = pid_on_port(TTS_PORT)
        tts_gpu = gpu_memory_for_pid(tts_pid)
        threshold = int(settings.get("tts_restart_threshold_mb") or 0)
        if threshold and tts_gpu > threshold:
            restart_tts_async(f"当前 {tts_gpu} MB，阈值 {threshold} MB")
        return audio
    except Exception as exc:
        mark_call("tts", "失败", started, error=exc)
        raise


def play_audio_async(audio, label="播放"):
    if not audio:
        mark_call("audio", "失败", error="没有可播放的音频")
        return False

    def play():
        started_audio = time.time()
        mark_call("audio", "播放中", started_audio)
        try:
            play_audio(audio)
            mark_call("audio", "完成", started_audio)
        except Exception as exc:
            mark_call("audio", "失败", started_audio, error=exc)

    threading.Thread(target=play, daemon=True).start()
    return True


def chat(message):
    global last_audio
    ensure_base_services(False)
    started = time.time()
    mark_call("llm", "生成中", started)
    try:
        reply = chat_with_agent(message)
        mark_call("llm", "完成", started)
    except Exception as exc:
        mark_call("llm", "失败", started, error=exc)
        raise
    audio = b""
    if settings["speak"]:
        ready, reason = tts_ready()
        if ready:
            try:
                audio = synthesize_with_policy(reply)
                with state_lock:
                    last_audio = audio
                if audio and settings.get("local_playback", True):
                    play_audio_async(audio)
            except Exception as exc:
                log(f"本轮语音生成失败，已保留文字回复：{exc}")
        else:
            mark_call("tts", "待调用", error=reason, extra={"last_bytes": 0})
            log(f"跳过本轮语音：{reason}")
    return reply, audio


def gpu_process_memory_map():
    now = time.time()
    with state_lock:
        if now - gpu_process_cache["at"] < 5:
            return dict(gpu_process_cache["memory"])

    memory = {}
    proc = run_capture([
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$samples=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples;"
            "$samples | Where-Object { $_.CookedValue -gt 0 -and $_.InstanceName -match 'pid_(\\d+)_' } | "
            "ForEach-Object { \"$($matches[1]),$([int64]$_.CookedValue)\" }"
        ),
    ], timeout=4)
    if proc and proc.returncode == 0:
        bytes_by_pid = {}
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                used_bytes = int(parts[1])
            except ValueError:
                continue
            bytes_by_pid[pid] = bytes_by_pid.get(pid, 0) + used_bytes
        for pid, used_bytes in bytes_by_pid.items():
            memory[pid] = max(memory.get(pid, 0), int(round(used_bytes / 1024 / 1024)))
    if memory:
        return memory

    proc = run_capture([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], timeout=6)
    if proc and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                used = int(float(parts[-1]))
            except ValueError:
                continue
            memory[pid] = max(memory.get(pid, 0), used)
    with state_lock:
        gpu_process_cache["at"] = time.time()
        gpu_process_cache["memory"] = dict(memory)
    return memory


def process_working_set_mb(pid):
    if not pid:
        return 0
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
        if not handle:
            return 0
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        ctypes.windll.kernel32.CloseHandle(handle)
        if not ok:
            return 0
        return round(counters.WorkingSetSize / 1024 / 1024, 1)
    except Exception:
        return 0


def gpu_memory_for_pid(pid):
    if not pid:
        return 0
    return gpu_process_memory_map().get(pid, 0)


def service_status():
    llm_pid = pid_on_port(LLM_PORT)
    tts_pid = pid_on_port(TTS_PORT) or pid_by_script("chatterbox_cuda_server.py")
    tts_health = None
    if tts_pid:
        tts_health = get_tts_health() or {"status": "unknown"}
    tts_load = get_tts_load_status()
    if tts_health and tts_health.get("status") == "ok":
        tts_load = {**tts_load, "status": "ok", "percent": 100, "step": "语音模型已就绪"}
    gpu_memory = gpu_process_memory_map()
    return {
        "llm": {
            "running": llm_pid is not None,
            "pid": llm_pid,
            "gpu_memory_mb": gpu_memory.get(llm_pid, 0) if llm_pid else 0,
            "ram_mb": process_working_set_mb(llm_pid),
        },
        "tts": {
            "running": tts_pid is not None,
            "pid": tts_pid,
            "health": tts_health,
            "load": tts_load,
            "gpu_memory_mb": gpu_memory.get(tts_pid, 0) if tts_pid else 0,
            "ram_mb": process_working_set_mb(tts_pid),
        },
    }


def memory_status():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "load": stat.dwMemoryLoad,
        "total_gb": round(stat.ullTotalPhys / 1024 ** 3, 2),
        "avail_gb": round(stat.ullAvailPhys / 1024 ** 3, 2),
    }


def gpu_status():
    now = time.time()
    with state_lock:
        if now - gpu_status_cache["at"] < 5:
            return dict(gpu_status_cache["status"])

    proc = run_capture([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], timeout=6)
    if not proc or proc.returncode != 0 or not proc.stdout.strip():
        status = {"available": False}
        with state_lock:
            gpu_status_cache["at"] = time.time()
            gpu_status_cache["status"] = dict(status)
        return status
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    status = {
        "available": True,
        "memory_used_mb": int(float(parts[0])),
        "memory_total_mb": int(float(parts[1])),
        "util": int(float(parts[2])),
        "temp": int(float(parts[3])),
        "power": parts[4],
    }
    with state_lock:
        gpu_status_cache["at"] = time.time()
        gpu_status_cache["status"] = dict(status)
    return status


def full_status():
    with state_lock:
        return {
            "services": service_status(),
            "gpu": gpu_status(),
            "memory": memory_status(),
            "settings": dict(settings),
            "model_options": discover_model_options(),
            "calls": json.loads(json.dumps(call_state, ensure_ascii=False)),
            "audio_ready": bool(last_audio),
            "logs": list(logs[-80:]),
            "history_len": len(conversation_history),
            "persona": SYSTEM_PROMPT,
        }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>控制台</title>
<style>
  :root { color-scheme: dark; font-family:"Microsoft YaHei UI","Segoe UI",system-ui,sans-serif; background:#101214; color:#e7e1d7; }
  html, body { width:100%; height:100%; overflow:hidden; }
  body { margin:0; display:grid; grid-template-columns:520px minmax(360px,1fr) 360px; }
  aside, main, .personaPane { height:100vh; box-sizing:border-box; overflow:hidden; }
  aside { border-right:1px solid #2a2f33; padding:12px; background:#15191c; display:grid; grid-template-rows:auto auto minmax(0,1fr); gap:10px; }
  main { display:grid; grid-template-rows:46px minmax(0,1fr) 52px 58px; min-width:0; }
  .personaPane { border-left:1px solid #2a2f33; padding:12px; background:#15191c; display:grid; grid-template-rows:auto minmax(0,1fr) auto auto; gap:10px; }
  h1 { font-size:18px; margin:0; }
  h2 { font-size:12px; color:#aeb7bd; margin:0 0 6px; letter-spacing:0; }
  button, select, input, textarea { background:#22282d; color:#f1eee8; border:1px solid #3b444b; border-radius:6px; padding:6px 8px; min-width:0; box-sizing:border-box; font:inherit; }
  button { cursor:pointer; white-space:nowrap; height:32px; }
  button:hover { background:#2d353b; }
  button.danger { background:#3a2526; border-color:#704244; color:#ffe6e6; }
  button.danger:hover { background:#4a2b2d; }
  label { display:grid; gap:3px; font-size:12px; color:#bec5ca; margin:0; min-width:0; }
  input, select { width:100%; height:31px; }
  textarea { width:100%; height:100%; resize:none; line-height:1.45; font-size:13px; }
  .topbar { display:flex; align-items:center; gap:10px; }
  .topbar .pill { margin-left:auto; color:#aeb7bd; font-size:12px; }
  .panelGrid { min-height:0; display:grid; grid-template-columns:1fr 1fr; gap:10px; overflow:hidden; }
  .section { min-width:0; background:#1b2024; border:1px solid #2a3136; border-radius:8px; padding:9px; display:grid; gap:8px; align-content:start; }
  .monitor { grid-template-rows:auto auto auto auto minmax(0,1fr); align-content:stretch; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
  .grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; }
  .row { display:flex; gap:8px; align-items:center; margin:0; }
  .row input[type=checkbox] { width:auto; height:auto; }
  .metric { background:#121619; border:1px solid #2a3136; border-radius:7px; padding:7px; font-size:12px; min-width:0; }
  .metric strong { color:#f1eee8; font-weight:600; }
  .bar { height:7px; background:#30383e; border-radius:99px; overflow:hidden; margin-top:6px; }
  .fill { height:100%; width:0%; background:#b6d36b; }
  .loadFill { background:#e9c46a; transition:width .25s ease; }
  .stackBar { height:10px; position:relative; }
  .seg { position:absolute; top:0; bottom:0; width:0%; }
  .seg.llm { background:#83c5be; }
  .seg.tts { background:#e9c46a; }
  .seg.other { background:#6f7b83; }
  .legend { margin-top:6px; display:flex; flex-wrap:wrap; gap:6px 10px; color:#aeb7bd; font-size:11px; }
  .legend span::before { content:""; display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:4px; vertical-align:-1px; background:#6f7b83; }
  .legend .llm::before { background:#83c5be; }
  .legend .tts::before { background:#e9c46a; }
  .legend .other::before { background:#6f7b83; }
  .state { display:grid; grid-template-columns:72px 1fr; gap:4px 8px; font-size:12px; color:#c8d0d4; }
  .state span:nth-child(odd) { color:#8d969c; }
  .logBox { min-height:0; overflow:auto; background:#101214; border:1px solid #2a2f33; border-radius:8px; padding:8px; font-size:11px; white-space:pre-wrap; }
  #chat { min-height:0; padding:14px; overflow:auto; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:760px; padding:10px 12px; border-radius:8px; line-height:1.45; white-space:pre-wrap; }
  .user { align-self:flex-end; background:#26333e; }
  .bot { align-self:flex-start; background:#24211e; border:1px solid #383029; }
  header { padding:10px 14px; border-bottom:1px solid #2a2f33; display:flex; gap:10px; align-items:center; }
  footer { padding:10px 14px; border-top:1px solid #2a2f33; display:flex; gap:10px; }
  footer input { flex:1; }
  .status { font-size:13px; color:#aeb7bd; }
  .audioBar { border-top:1px solid #2a2f33; padding:8px 14px; display:flex; gap:10px; align-items:center; }
  .audioBar audio { width:min(520px,100%); height:34px; }
  .personaActions { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .hint { font-size:12px; color:#9aa4aa; line-height:1.45; }
  @media (max-width:1180px) { body { grid-template-columns:500px minmax(340px,1fr) 320px; } }
</style>
</head>
<body>
<aside>
  <div class="topbar"><h1>控制台</h1><span class="pill" id="svcSummary">正在读取状态</span></div>
  <div class="grid2">
    <button onclick="service('llm','start')">启动大模型</button>
    <button onclick="service('llm','stop')">停止大模型</button>
    <button onclick="service('tts','start')">启动语音</button>
    <button onclick="service('tts','stop')">停止语音</button>
    <button class="danger" onclick="shutdownAll()" style="grid-column:1 / -1">关闭全部并退出</button>
  </div>
  <div class="panelGrid">
    <section class="section">
      <h2>大模型设置</h2>
      <label>模型文件 <select id="modelSelect"></select></label>
      <div class="grid2">
        <label>GPU 层数 <input id="layers" type="number" min="0" max="99" value="99"></label>
        <label>上下文长度 <input id="ctx" type="number" min="512" step="512" value="4096"></label>
      </div>
      <div class="grid3">
        <label>温度 <input id="llmTemp" type="number" min="0" max="2" step="0.05" value="0.8"></label>
        <label>Top P <input id="llmTopP" type="number" min="0.05" max="1" step="0.05" value="0.9"></label>
        <label>最大回复 <input id="llmMaxTokens" type="number" min="64" max="4096" step="64" value="512"></label>
      </div>
      <label>保留消息数 <input id="historyMessages" type="number" min="6" max="60" step="2" value="24"></label>
      <h2>语音设置</h2>
      <div class="grid3">
        <label>模式 <select id="ttsMode"><option value="gpu">GPU</option></select></label>
        <label>设备 <select id="ttsDevice"><option value="cuda">CUDA</option></select></label>
        <label>语言 <select id="ttsLanguage"><option value="zh">中文</option><option value="ja">日语</option><option value="en">英语</option><option value="z">自动</option></select></label>
      </div>
      <div class="grid3">
        <label>温度 <input id="ttsTemp" type="number" min="0.1" max="2" step="0.05" value="0.8"></label>
        <label>Top P <input id="ttsTopP" type="number" min="0.05" max="1" step="0.05" value="0.95"></label>
        <label>最大 token <input id="ttsMaxTokens" type="number" min="0" max="1024" step="20" value="0"></label>
      </div>
      <div class="grid2">
        <label>显存上限 MB <input id="ttsMemLimit" type="number" min="1200" max="4096" step="100" value="2300"></label>
        <label>生成上限 <input id="ttsGenCap" type="number" min="80" max="420" step="20" value="220"></label>
      </div>
      <label>回收阈值 MB <input id="ttsRestartLimit" type="number" min="1600" max="4096" step="100" value="2600"></label>
      <div class="grid2">
        <label>情绪强度 <input id="ttsExaggeration" type="number" min="0" max="2" step="0.05" value="0.5"></label>
        <label>重复惩罚 <input id="ttsRepeat" type="number" min="1" max="2" step="0.05" value="1.2"></label>
      </div>
      <div class="grid2">
        <label>声线 <select id="voice"><option value="female">女声</option><option value="male">男声</option></select></label>
        <div class="grid2">
          <label class="row"><input id="speak" type="checkbox" checked>生成语音</label>
          <label class="row"><input id="localPlayback" type="checkbox" checked>本机播放</label>
        </div>
      </div>
      <button onclick="saveSettings()">保存设置</button>
    </section>
    <section class="section monitor">
      <h2>资源监控</h2>
      <div class="metric">
        <div id="gpuText">GPU</div>
        <div class="bar stackBar">
          <div id="gpuLlmSeg" class="seg llm"></div>
          <div id="gpuTtsSeg" class="seg tts"></div>
          <div id="gpuOtherSeg" class="seg other"></div>
        </div>
        <div id="gpuLegend" class="legend"></div>
      </div>
      <div class="metric"><div id="ramText">内存</div><div class="bar"><div id="ramFill" class="fill"></div></div></div>
      <div class="metric" id="svcText">服务</div>
      <div class="metric">
        <div id="ttsLoadText">语音加载：未启动</div>
        <div class="bar"><div id="ttsLoadFill" class="fill loadFill"></div></div>
      </div>
      <h2>模型调用状态</h2>
      <div class="metric state" id="callText"></div>
      <h2>运行日志</h2>
      <div class="logBox" id="logs"></div>
    </section>
  </div>
</aside>
<main>
  <header><strong>对话</strong><span id="busy" class="status"></span></header>
  <section id="chat"></section>
  <div class="audioBar"><span class="status">音频</span><audio id="audioPlayer" controls></audio></div>
  <footer>
    <input id="message" placeholder="输入消息..." onkeydown="if(event.key==='Enter') sendMessage()">
    <button onclick="sendMessage()">发送</button>
  </footer>
</main>
<section class="personaPane">
  <div class="topbar"><h1>角色设定</h1><span class="pill" id="personaState">未修改</span></div>
  <textarea id="personaEditor" spellcheck="false"></textarea>
  <div class="personaActions">
    <button onclick="savePersona()">保存角色</button>
    <button onclick="resetPersona()">恢复默认</button>
  </div>
  <div class="personaActions">
    <button onclick="clearHistory()">清空对话</button>
    <button onclick="refresh()">刷新状态</button>
  </div>
  <div class="hint">保存角色会立即更新 system prompt，并保留最近几条对话。若出现 400，系统会自动缩短上下文重试。</div>
</section>
<script>
let audioUnlocked=false;
const player=new Audio();
let lastAudioB64='';
let personaDirty=false;
let modelDirty=false;
let settingsDirty=false;
let lastLogText='';
let refreshInFlight=false;
function audioEl(){ return document.getElementById('audioPlayer'); }
async function unlockAudio(){
  if(audioUnlocked) return;
  try{ player.src='data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA='; await player.play(); player.pause(); audioUnlocked=true; }
  catch(e){ audioUnlocked=true; }
}
async function api(path, body){
  const opt=body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r=await fetch(path,opt); const data=await r.json(); if(!r.ok) throw new Error(data.error||r.statusText); return data;
}
function addMsg(text, cls){ const el=document.createElement('div'); el.className='msg '+cls; el.textContent=text; chat.appendChild(el); chat.scrollTop=chat.scrollHeight; }
async function sendMessage(){
  await unlockAudio();
  const input=document.getElementById('message'); const text=input.value.trim(); if(!text) return;
  input.value=''; addMsg(text,'user'); busy.textContent='正在生成回复...';
  try{
    const data=await api('/api/chat',{message:text});
    addMsg(data.reply,'bot');
    if(data.audio_b64){
      lastAudioB64=data.audio_b64;
      const src='data:audio/wav;base64,'+data.audio_b64;
      player.src=src; audioEl().src=src;
      if(!localPlayback.checked){ try{ await audioEl().play(); }catch(e){ busy.textContent='浏览器阻止了自动播放'; } }
      else { busy.textContent='已由本机播放'; }
    }
  }catch(e){ addMsg('[错误] '+e.message,'bot'); }
  busy.textContent=''; refresh();
}
async function service(name, action){
  const modelName=name==='llm'?'大模型':'语音模型';
  busy.textContent=(action==='start'?'正在启动':'正在停止')+modelName+'...';
  try{ await api('/api/service',{name,action}); }catch(e){ alert(e.message); }
  busy.textContent=''; refresh();
}
async function saveSettings(){
  await api('/api/settings',{model_path:modelSelect.value,ctx_size:+ctx.value,n_gpu_layers:+layers.value,llm_temperature:+llmTemp.value,llm_top_p:+llmTopP.value,llm_max_tokens:+llmMaxTokens.value,history_messages:+historyMessages.value,tts_mode:ttsMode.value,tts_device:ttsDevice.value,tts_language:ttsLanguage.value,tts_temperature:+ttsTemp.value,tts_top_p:+ttsTopP.value,tts_exaggeration:+ttsExaggeration.value,tts_repetition_penalty:+ttsRepeat.value,tts_max_new_tokens:+ttsMaxTokens.value,tts_cuda_mem_limit_mb:+ttsMemLimit.value,tts_generation_cap:+ttsGenCap.value,tts_restart_threshold_mb:+ttsRestartLimit.value,voice:voice.value,speak:speak.checked,local_playback:localPlayback.checked});
  modelDirty=false;
  settingsDirty=false;
  busy.textContent='设置已保存'; setTimeout(()=>busy.textContent='',1200); refresh();
}
async function savePersona(){
  personaState.textContent='正在保存...';
  try{ const data=await api('/api/persona',{persona:personaEditor.value}); personaEditor.value=data.persona; personaDirty=false; personaState.textContent='已保存'; }
  catch(e){ personaState.textContent='保存失败'; alert(e.message); }
}
async function resetPersona(){
  if(!confirm('恢复默认角色设定？')) return;
  const data=await api('/api/persona/reset',{});
  personaEditor.value=data.persona; personaDirty=false; personaState.textContent='已恢复默认';
}
async function clearHistory(){
  await api('/api/history/clear',{});
  chat.innerHTML=''; busy.textContent='对话已清空'; setTimeout(()=>busy.textContent='',1200); refresh();
}
async function shutdownAll(){
  if(!confirm('关闭大模型、语音模型和控制台？')) return;
  busy.textContent='正在关闭全部...';
  try{ await api('/api/shutdown',{}); }
  catch(e){ busy.textContent='关闭请求已发送'; }
}
function callLine(label,c){
  const err=c.last_error ? ` 错误：${c.last_error}` : '';
  const bytes=c.last_bytes ? `，${Math.round(c.last_bytes/1024)} KB` : '';
  return `<span>${label}</span><strong>${c.state}</strong><span>耗时</span><span>${c.last_ms||0} ms${bytes}</span><span>时间</span><span>${c.last_at||'-'}${err}</span>`;
}
function modelLine(label, item, extra=''){
  if(!item || !item.running) return `${label}：已停止`;
  const gpu = item.gpu_memory_mb ? `${item.gpu_memory_mb} MB` : '未知';
  const ram = item.ram_mb ? `${item.ram_mb} MB` : '-';
  return `${label}：PID ${item.pid}，显存 ${gpu}，内存 ${ram}${extra}`;
}
function ttsDisplayState(s, health, load){
  if(!s.services.tts.running) return s.settings.speak ? '待自动加载' : '已停止';
  if(health.status==='ok') return '运行中';
  if(load.status==='error') return '加载失败';
  if(load.status==='starting' || load.status==='loading') return '加载中';
  return '未就绪';
}
function ttsModelLine(item, state, extra=''){
  if(!item || !item.running) return `语音：${state}`;
  const gpu = item.gpu_memory_mb ? `${item.gpu_memory_mb} MB` : '未知';
  const ram = item.ram_mb ? `${item.ram_mb} MB` : '-';
  return `语音：${state}，PID ${item.pid}，显存 ${gpu}，内存 ${ram}${extra}`;
}
function setGpuSeg(el,left,width){
  el.style.left=Math.max(0,Math.min(100,left))+'%';
  el.style.width=Math.max(0,Math.min(100,width))+'%';
}
function fmtMb(v){ return v ? `${v} MB` : '0 MB'; }
function syncModelOptions(options, selected){
  const signature=JSON.stringify((options||[]).map(o=>[o.path,o.label]));
  if(modelSelect.dataset.signature!==signature){
    modelSelect.innerHTML='';
    for(const item of options||[]){
      const opt=document.createElement('option');
      opt.value=item.path; opt.textContent=item.label;
      modelSelect.appendChild(opt);
    }
    modelSelect.dataset.signature=signature;
  }
  if(selected && !modelDirty) modelSelect.value=selected;
}
async function refresh(){
  if(refreshInFlight) return;
  refreshInFlight=true;
  try{
  const s=await api('/api/status'); const g=s.gpu, m=s.memory;
  if(g.available){
    const llmGpu=s.services.llm.gpu_memory_mb||0;
    const ttsGpu=s.services.tts.gpu_memory_mb||0;
    const usedGpu=g.memory_used_mb||0;
    const totalGpu=g.memory_total_mb||1;
    const otherGpu=Math.max(0,usedGpu-llmGpu-ttsGpu);
    const llmPct=100*llmGpu/totalGpu;
    const ttsPct=100*ttsGpu/totalGpu;
    const otherPct=100*otherGpu/totalGpu;
    gpuText.textContent=`GPU：${usedGpu}/${g.memory_total_mb} MB，占用 ${g.util}%，${g.temp}°C`;
    setGpuSeg(gpuLlmSeg,0,llmPct);
    setGpuSeg(gpuTtsSeg,llmPct,ttsPct);
    setGpuSeg(gpuOtherSeg,llmPct+ttsPct,otherPct);
    gpuLegend.innerHTML=`<span class="llm">大模型 ${fmtMb(llmGpu)}</span><span class="tts">语音 ${fmtMb(ttsGpu)}</span><span class="other">其它 ${fmtMb(otherGpu)}</span>`;
  }
  else {
    gpuText.textContent='GPU：未检测到';
    setGpuSeg(gpuLlmSeg,0,0); setGpuSeg(gpuTtsSeg,0,0); setGpuSeg(gpuOtherSeg,0,0);
    gpuLegend.textContent='';
  }
  ramText.textContent=`内存：${(m.total_gb-m.avail_gb).toFixed(2)}/${m.total_gb} GB（${m.load}%）`; ramFill.style.width=m.load+'%';
  const ttsHealth=s.services.tts.health||{};
  const ttsLoad=s.services.tts.load||{};
  const ttsState=ttsDisplayState(s,ttsHealth,ttsLoad);
  svcSummary.textContent=`大模型 ${s.services.llm.running?'运行中':'已停止'} · 语音 ${ttsState}`;
  const ttsExtra = s.services.tts.running ? `，设备 ${ttsHealth.device||'cuda'}，状态 ${ttsState}` : (s.settings.speak ? '，待自动加载' : '，已手动关闭语音生成');
  svcText.textContent=`${modelLine('大模型',s.services.llm)}；${ttsModelLine(s.services.tts,ttsState,ttsExtra)}；历史：${s.history_len} 条`;
  const loadPct=Math.max(0,Math.min(100,Number(ttsLoad.percent||0)));
  const loadStep=ttsLoad.step||'语音模型未启动';
  const loadState=ttsLoad.status||'idle';
  ttsLoadText.textContent=`语音加载：${loadPct}% · ${loadStep}${loadState==='error'&&ttsLoad.error?' · '+ttsLoad.error:''}`;
  ttsLoadFill.style.width=loadPct+'%';
  callText.innerHTML=callLine('大模型',s.calls.llm)+callLine('语音',s.calls.tts)+callLine('播放',s.calls.audio);
  const nextLogs=s.logs.join('\n');
  const shouldFollowLogs=logs.scrollTop+logs.clientHeight>=logs.scrollHeight-12;
  if(nextLogs!==lastLogText){
    logs.textContent=nextLogs;
    lastLogText=nextLogs;
    if(shouldFollowLogs) logs.scrollTop=logs.scrollHeight;
  }
  if(!settingsDirty){
    layers.value=s.settings.n_gpu_layers; ctx.value=s.settings.ctx_size; historyMessages.value=s.settings.history_messages;
    syncModelOptions(s.model_options,s.settings.model_path);
    llmTemp.value=s.settings.llm_temperature; llmTopP.value=s.settings.llm_top_p; llmMaxTokens.value=s.settings.llm_max_tokens;
    ttsMode.value=s.settings.tts_mode; ttsDevice.value=s.settings.tts_device; ttsLanguage.value=s.settings.tts_language;
    ttsTemp.value=s.settings.tts_temperature; ttsTopP.value=s.settings.tts_top_p; ttsExaggeration.value=s.settings.tts_exaggeration;
    ttsRepeat.value=s.settings.tts_repetition_penalty; ttsMaxTokens.value=s.settings.tts_max_new_tokens;
    ttsMemLimit.value=s.settings.tts_cuda_mem_limit_mb; ttsGenCap.value=s.settings.tts_generation_cap; ttsRestartLimit.value=s.settings.tts_restart_threshold_mb;
    voice.value=s.settings.voice; speak.checked=s.settings.speak; localPlayback.checked=s.settings.local_playback;
  }
  if(!personaDirty && document.activeElement!==personaEditor){ personaEditor.value=s.persona||''; personaState.textContent='已同步'; }
  } finally {
    refreshInFlight=false;
  }
}
personaEditor.addEventListener('input',()=>{ personaDirty=true; personaState.textContent='未保存'; });
for(const el of [modelSelect,layers,ctx,llmTemp,llmTopP,llmMaxTokens,historyMessages,ttsMode,ttsDevice,ttsLanguage,ttsTemp,ttsTopP,ttsMaxTokens,ttsMemLimit,ttsGenCap,ttsRestartLimit,ttsExaggeration,ttsRepeat,voice,speak,localPlayback]){
  el.addEventListener('input',()=>{ settingsDirty=true; });
  el.addEventListener('change',()=>{ settingsDirty=true; });
}
modelSelect.addEventListener('change',()=>{ modelDirty=true; });
setInterval(refresh,1500); refresh();
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            self.send_json(200, full_status())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        try:
            if self.path == "/api/settings":
                data = self.read_json()
                old_restart_values = {key: settings.get(key) for key in ("model_path", "ctx_size", "n_gpu_layers")}
                with state_lock:
                    for key in (
                        "model_path", "ctx_size", "n_gpu_layers", "llm_temperature", "llm_top_p", "llm_max_tokens",
                        "tts_mode", "tts_device", "tts_language", "tts_temperature", "tts_top_p",
                        "tts_exaggeration", "tts_repetition_penalty", "tts_max_new_tokens",
                        "tts_cuda_mem_limit_mb", "tts_generation_cap", "tts_restart_threshold_mb",
                        "speak", "local_playback", "voice", "speed", "history_messages"
                    ):
                        if key in data:
                            settings[key] = data[key]
                    if "model_path" in data:
                        model_path = Path(str(settings["model_path"]))
                        if not model_path.is_absolute():
                            model_path = ROOT / model_path
                        settings["model_path"] = str(model_path)
                    settings["tts_mode"] = "gpu"
                    settings["tts_device"] = "cuda"
                    save_settings()
                log("设置已更新")
                if any(settings.get(key) != old_restart_values.get(key) for key in ("model_path", "ctx_size", "n_gpu_layers")):
                    if pid_on_port(LLM_PORT):
                        restart_llm_async("设置已保存")
                self.send_json(200, {"ok": True, "settings": settings})
                return
            if self.path == "/api/persona":
                data = self.read_json()
                text = str(data.get("persona", "")).strip()
                if not text:
                    raise ValueError("角色设定不能为空")
                with state_lock:
                    set_persona(text, persist=True)
                log("角色设定已保存")
                self.send_json(200, {"ok": True, "persona": SYSTEM_PROMPT})
                return
            if self.path == "/api/persona/reset":
                with state_lock:
                    set_persona(DEFAULT_PERSONA, persist=True)
                log("角色设定已恢复默认")
                self.send_json(200, {"ok": True, "persona": SYSTEM_PROMPT})
                return
            if self.path == "/api/history/clear":
                with state_lock:
                    reset_conversation(keep_recent=False)
                log("对话历史已清空")
                self.send_json(200, {"ok": True})
                return
            if self.path == "/api/shutdown":
                self.send_json(200, {"ok": True})
                shutdown_all_async()
                return
            if self.path == "/api/service":
                data = self.read_json()
                name, action = data.get("name"), data.get("action")
                if name == "llm" and action == "start":
                    ok = start_llm()
                elif name == "llm" and action == "stop":
                    ok = stop_llm()
                elif name == "tts" and action == "start":
                    ok = start_tts("cuda")
                elif name == "tts" and action == "stop":
                    ok = stop_tts()
                else:
                    raise ValueError("未知服务操作")
                self.send_json(200, {"ok": ok})
                return
            if self.path == "/api/chat":
                message = self.read_json().get("message", "")
                if not message:
                    raise ValueError("消息不能为空")
                reply, audio = chat(message)
                self.send_json(200, {
                    "reply": reply,
                    "audio_b64": base64.b64encode(audio).decode("ascii") if audio else "",
                })
                return
            if self.path == "/api/replay":
                with state_lock:
                    audio = last_audio
                if not audio:
                    raise ValueError("没有可重播音频")
                play_audio_async(audio, "重播")
                self.send_json(200, {"ok": True})
                return
            self.send_response(404)
            self.end_headers()
        except Exception as exc:
            log(f"错误：{exc}")
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        pass


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    for old_pid in pids_on_port(PANEL_PORT):
        if old_pid != os.getpid():
            stop_pid(old_pid)
    time.sleep(0.5)
    log(f"控制面板已启动：http://{HOST}:{PANEL_PORT}")
    if os.environ.get("AGENT_AUTOSTART") == "1":
        autostart_services_async()
    ReusableThreadingHTTPServer((HOST, PANEL_PORT), Handler).serve_forever()
