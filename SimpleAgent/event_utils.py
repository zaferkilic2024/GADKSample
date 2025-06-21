from typing import Any
from datetime import datetime

def format_timestamp(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Invalid timestamp"

def pretty_event_print(event: Any):
    print("\n🟡 [Event Debug Info]")
    print(f"📌 ID: {getattr(event, 'id', 'N/A')}")
    print(f"👤 Author: {getattr(event, 'author', 'N/A')}")

    raw_ts = getattr(event, 'timestamp', None)
    if raw_ts is not None:
        friendly_ts = format_timestamp(raw_ts)
        print(f"⏰ Timestamp: {raw_ts} → 📅 {friendly_ts}")
    else:
        print("⏰ Timestamp: N/A")

    print(f"✅ Final Response: {event.is_final_response() if hasattr(event, 'is_final_response') else 'N/A'}")
    print("🧩 Parts:")

    content = getattr(event, 'content', None)
    if content and hasattr(content, 'parts'):
        for i, part in enumerate(content.parts):
            print(f"  ├─ Part {i + 1}:")
            if part.text:
                print(f"  │   📝 Text: {part.text}")
            if part.function_call:
                fc = part.function_call
                print(f"  │   📞 Function Call: {fc.name} | Args: {fc.args}")
            if part.function_response:
                fr = part.function_response
                print(f"  │   🎯 Function Response: {fr.name} | Response: {fr.response}")
    else:
        print("  └─ No content parts found.")

    print("🟩 End Event\n")

def handle_event_response(event: Any):
    """Tüm event analizini ve response işlemini üstlenen soyut fonksiyon."""
    pretty_event_print(event)

    if hasattr(event, "is_final_response") and event.is_final_response():
        content = getattr(event, "content", None)
        if content and hasattr(content, "parts") and content.parts:
            first_part = content.parts[0]
            if first_part and hasattr(first_part, "text") and first_part.text:
                print(f"🤖 [Agent] {event.author}:\n{first_part.text}")
