#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("prompt_id", ""), row.get("model", ""), row.get("wav", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda r: (r.get("prompt_id", ""), r.get("model", "")))
    return rows


def rel_audio_path(wav: str, output_path: Path) -> str:
    wav_path = Path(wav)
    if not wav_path.is_absolute():
        wav_path = (Path.cwd() / wav_path).resolve()
    return Path(html.escape(str(wav_path.relative_to(output_path.parent.resolve())))).as_posix()


def render_html(rows: list[dict[str, str]], output_path: Path, title: str) -> str:
    by_prompt: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_prompt.setdefault(row.get("prompt_id", ""), []).append(row)

    prompt_sections: list[str] = []
    for prompt_id, prompt_rows in by_prompt.items():
        text = prompt_rows[0].get("text", "")
        cards: list[str] = []
        for row in prompt_rows:
            key = f"{row.get('prompt_id', '')}:{row.get('model', '')}"
            audio_src = rel_audio_path(row.get("wav", ""), output_path)
            cards.append(
                f"""
          <article class="sample" data-key="{html.escape(key)}">
            <header>
              <h3>{html.escape(row.get("model", ""))}</h3>
              <span>{html.escape(row.get("duration_sec", ""))}s</span>
            </header>
            <audio controls preload="metadata" src="{audio_src}"></audio>
            <dl>
              <dt>Peak</dt><dd>{html.escape(row.get("peak_abs", ""))}</dd>
              <dt>WAV</dt><dd>{html.escape(Path(row.get("wav", "")).name)}</dd>
            </dl>
            <label>Score
              <select data-field="score">
                <option value=""></option>
                <option value="5">5 excellent</option>
                <option value="4">4 good</option>
                <option value="3">3 usable</option>
                <option value="2">2 weak</option>
                <option value="1">1 poor</option>
              </select>
            </label>
            <label>Notes
              <textarea data-field="notes" rows="3"></textarea>
            </label>
          </article>
"""
            )
        prompt_sections.append(
            f"""
      <section class="prompt">
        <h2>{html.escape(prompt_id)} <span>{html.escape(text)}</span></h2>
        <div class="samples">
          {"".join(cards)}
        </div>
      </section>
"""
        )

    payload = json.dumps(
        [{"prompt_id": r.get("prompt_id", ""), "model": r.get("model", "")} for r in rows],
        ensure_ascii=True,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; background: #f6f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    p {{ margin: 0 0 18px; color: #57606a; }}
    .actions {{ display: flex; gap: 8px; margin: 16px 0 24px; }}
    button {{ border: 1px solid #d0d7de; background: #fff; border-radius: 6px; padding: 7px 11px; cursor: pointer; }}
    button:hover {{ background: #f3f4f6; }}
    .prompt {{ border-top: 1px solid #d8dee4; padding: 22px 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h2 span {{ display: block; margin-top: 4px; color: #57606a; font-size: 15px; font-weight: 500; }}
    .samples {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }}
    .sample {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }}
    .sample header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 8px; }}
    h3 {{ margin: 0; font-size: 14px; overflow-wrap: anywhere; }}
    header span {{ color: #57606a; font-size: 13px; }}
    audio {{ width: 100%; height: 34px; }}
    dl {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 8px; margin: 10px 0; font-size: 12px; color: #57606a; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    label {{ display: grid; gap: 5px; margin-top: 9px; font-size: 13px; color: #57606a; }}
    select, textarea {{ width: 100%; box-sizing: border-box; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px; font: inherit; }}
    textarea {{ resize: vertical; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p>Listen to each model for the same prompt, score naturalness/intelligibility, then export notes.</p>
    <div class="actions">
      <button id="export">Export JSON</button>
      <button id="clear">Clear saved notes</button>
    </div>
    {"".join(prompt_sections)}
  </main>
  <script>
    const rows = {payload};
    const storageKey = "kikuyu-tts-listening:" + location.pathname;
    const state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    document.querySelectorAll(".sample").forEach((sample) => {{
      const key = sample.dataset.key;
      const saved = state[key] || {{}};
      sample.querySelectorAll("[data-field]").forEach((field) => {{
        field.value = saved[field.dataset.field] || "";
        field.addEventListener("input", () => {{
          state[key] = state[key] || {{}};
          state[key][field.dataset.field] = field.value;
          localStorage.setItem(storageKey, JSON.stringify(state));
        }});
      }});
    }});
    document.getElementById("export").addEventListener("click", () => {{
      const data = rows.map((row) => {{
        const key = row.prompt_id + ":" + row.model;
        return {{...row, ...(state[key] || {{}})}};
      }});
      const blob = new Blob([JSON.stringify(data, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "tts_listening_scores.json";
      a.click();
      URL.revokeObjectURL(url);
    }});
    document.getElementById("clear").addEventListener("click", () => {{
      localStorage.removeItem(storageKey);
      location.reload();
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML listening sheet for local TTS comparison WAVs.")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Kikuyu TTS Listening Sheet")
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    if not rows:
        raise ValueError("No rows found in manifests.")
    output_path.write_text(render_html(rows, output_path, args.title), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
