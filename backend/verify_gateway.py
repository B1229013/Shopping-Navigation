"""Smoke test for the CGU gateway — run BEFORE building any UI.

Confirms one OpenAI chat model (gpt-5.4-mini) and one local/Ollama model
(gpt-oss:20b) both answer through the *same* shared client, proving the single
key + base-URL override reaches both model families.

    cd backend
    python verify_gateway.py

Reads CGU_API_KEY from ../.env (loaded by server.config on import).
"""
import sys

from server.llm_gateway import chat, list_models, model_kind

PROBE_MODELS = ("gpt-5.4-mini", "gpt-oss:20b")


def main() -> int:
    try:
        models = list_models()
    except Exception as e:
        print(f"FAILED to reach /v1/models: {e}")
        print("  -> is CGU_API_KEY set in APPNAV-main/.env ?")
        return 1

    print(f"/v1/models -> {len(models)} models available")
    ok = True
    for m in PROBE_MODELS:
        if m not in models:
            print(f"  ! {m}: not in catalog (available e.g. {models[:3]}…)")
            ok = False
            continue
        try:
            out = chat(m, "Reply with exactly: ok", max_tokens=10, timeout=60)
            print(f"  [{model_kind(m)}] {m}: OK -> {out.strip()!r}")
        except Exception as e:
            print(f"  [{model_kind(m)}] {m}: FAILED -> {e}")
            ok = False

    print("\nRESULT:", "both models reachable — safe to build the UI" if ok
          else "one or more probes failed — fix before building the UI")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
