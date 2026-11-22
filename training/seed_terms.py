"""Operator-maintained vocabulary allowlist; excludes personal identifiers."""

SEEDS = [
    "nhận dạng giọng nói", "giọng nói", "phiên âm", "tiếng Việt", "phụ đề",
    "mô hình ngôn ngữ", "trí tuệ nhân tạo", "học máy", "học sâu",
    "mô hình", "tác vụ", "bộ điều phối", "độ trễ", "ngữ cảnh",
    "từ điển", "tối ưu hóa", "tự động hóa", "kiểm thử", "tích hợp",
    "kho tri thức", "dữ liệu", "truy vấn", "xác thực", "bảo mật",
    "SODA", "Google SODA", "ML Kit", "Live Transcribe", "SpeechRecognizer",
    "STT", "TTS", "ASR", "RNN-T", "TensorFlow Lite", "LiteRT",
    "ONNX", "PyTorch", "FastAPI", "WebSocket", "AudioWorklet",
    "PCM", "microphone", "resampling", "WER", "CER",
    "Chrome", "Chromium", "Android", "Windows", "Linux", "WSL2",
    "Google", "OpenAI", "Gemini", "Codex", "Claude", "LLM",
    "MCP", "MCP Device", "Palace", "CDPA", "RAG", "vector database",
    "retrieval", "embeddings", "fine-tuning", "LoRA",
    "Docker", "Kubernetes", "GitHub", "Git", "Python", "Go",
    "TypeScript", "JavaScript", "Kotlin", "Rust", "Gradle",
    "PostgreSQL", "SQLite", "Redis", "REST API", "GraphQL",
    "Cloudflare", "Tailscale", "ADB", "SSH", "RDP",
    "CI/CD", "DevOps", "unit test", "integration test",
    "API", "HTTP", "HTTPS", "OAuth", "JWT", "TLS",
    "batch processing", "streaming", "real-time", "edge inference",
]
# Exclude ambiguous single common words that could force ordinary speech.
EXCLUDE = {"Google", "mô hình", "dữ liệu", "tác vụ", "API", "HTTP", "HTTPS",
           "Go", "Git", "Chrome", "Windows", "Linux", "Python", "Android",
           "kiểm thử", "bảo mật", "tích hợp", "tự động hóa", "tối ưu hóa"}
TERMS = tuple(t for t in dict.fromkeys(SEEDS) if t not in EXCLUDE)