"""Diagnostic helper to list available models on the configured Groq account.

Usage:
    python -m scripts.list_models
"""

import os
import sys
from groq import Groq, AuthenticationError


def main() -> None:
    print("=" * 60)
    print("  VERDICT: Groq Available Models Diagnostic")
    print("=" * 60)

    # 1. Resolve API key from environment or app settings
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        try:
            from app.config import get_settings
            settings = get_settings()
            api_key = (
                settings.GROQ_API_KEY.get_secret_value()
                if hasattr(settings.GROQ_API_KEY, "get_secret_value")
                else str(settings.GROQ_API_KEY)
            ).strip()
        except Exception:
            pass

    if not api_key:
        print("\n[ERROR] No GROQ_API_KEY detected.")
        print("Please set GROQ_API_KEY in your .env file or environment:")
        print("    export GROQ_API_KEY=gsk_...")
        print("    # or on Windows PowerShell: $env:GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    # 2. Query Groq models API
    try:
        client = Groq(api_key=api_key)
        model_list = client.models.list()
        models = sorted([m.id for m in model_list.data])

        print(f"\nFound {len(models)} accessible models for your Groq account:\n")
        print(f"{'#':<4} {'Model ID':<45} {'Type'}")
        print("-" * 65)

        for idx, model_id in enumerate(models, start=1):
            category = "Chat / Completion"
            if "whisper" in model_id.lower():
                category = "Audio Transcription"
            elif "guard" in model_id.lower() or "safeguard" in model_id.lower():
                category = "Safety / Guardrail"
            elif "vision" in model_id.lower():
                category = "Multimodal Vision"

            prefix = "-> " if any(rec in model_id for rec in ["llama-3.3", "gpt-oss", "qwen"]) else "   "
            print(f"{prefix}{idx:<3} {model_id:<45} [{category}]")

        print("-" * 65)
        print("\nTip: Set GROQ_MODEL in your .env file to any of the above chat models.")
        print("     Default in VERDICT: llama-3.3-70b-versatile (or openai/gpt-oss-120b / openai/gpt-oss-20b)\n")

    except AuthenticationError:
        print("\n[ERROR] Groq authentication failed: Invalid API key.")
        print("Please check your GROQ_API_KEY in .env or console.groq.com.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to query Groq models API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
