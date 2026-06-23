"""
Gemini client for slab_v2 — Vertex AI service account auth (same pattern as
src/ai_floor_analyzer.py) plus structured-output JSON calls.

Every call uses response_mime_type=application/json with a response_schema,
so responses are parsed by the SDK, never by regex.
"""

from __future__ import annotations

import os
from pathlib import Path

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    load_dotenv()

    from google import genai
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")

    if not creds_path or not project:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT "
            "must be set in .env")

    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    _client = genai.Client(
        vertexai=True, project=project, location=location, credentials=creds)
    return _client


def get_model_name(override: str = "") -> str:
    if override:
        return override
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def call_gemini_json(
    prompt: str,
    images: list[bytes],
    response_schema: dict,
    model: str = "",
    log_path: str | None = None,
    tag: str = "",
    raw_path: str | None = None,
) -> dict:
    """One structured-output Gemini call. images = PNG bytes, in order,
    placed before the prompt text. Returns the parsed JSON dict.
    Raises RuntimeError on empty/invalid response.
    """
    import json
    import time
    from google.genai import types
    from google.genai import errors as genai_errors

    client = get_client()
    model_name = get_model_name(model)

    contents = [types.Part.from_bytes(data=b, mime_type="image/png")
                for b in images]
    contents.append(prompt)

    response = None
    last_err = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    seed=0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            break
        except genai_errors.APIError as e:
            last_err = e
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(15 * (attempt + 1))    # 15s, 30s, 45s backoff
                continue
            raise RuntimeError(f"Gemini API error ({tag}): {e}") from e
    if response is None:
        raise RuntimeError(f"Gemini API error ({tag}): {last_err}")
    raw = response.text or ""

    if raw_path:
        rp = Path(raw_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(raw, encoding="utf-8")

    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'=' * 70}\n[{tag}] model={model_name} "
                     f"images={len(images)}\n{'-' * 70}\nPROMPT:\n{prompt}\n"
                     f"{'-' * 70}\nRESPONSE:\n{raw}\n")

    if not raw.strip():
        raise RuntimeError(f"Gemini returned empty response ({tag})")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid JSON ({tag}): {e}") from e
