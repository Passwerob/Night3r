#!/usr/bin/env python3
"""Serve the Night3R review UI and persist Keep/Skip decisions."""

import argparse
import csv
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def parse_args():
    parser = argparse.ArgumentParser("Serve Night3R pair review")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17892)
    return parser.parse_args()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


class ReviewStore:
    def __init__(self, root):
        self.root = root.resolve()
        self.app_root = self.root / "review_app"
        self.selection_root = self.root / "review_selection"
        self.decisions_path = self.selection_root / "decisions.json"
        self.manifest_path = self.app_root / "review_manifest.json"
        self.lock = threading.Lock()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.items = {item["id"]: item for item in self.manifest["items"]}
        self.selection_root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self._sync_exports()

    def _load_state(self):
        if self.decisions_path.is_file():
            state = json.loads(self.decisions_path.read_text(encoding="utf-8"))
            state.setdefault("decisions", {})
            state.setdefault("history", [])
            return state
        return {"version": 1, "updated_at": now_iso(), "decisions": {}, "history": []}

    def _atomic_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _counts(self):
        values = [entry.get("decision") for entry in self.state["decisions"].values()]
        return {
            "keep": values.count("keep"),
            "skip": values.count("skip"),
            "unreviewed": len(self.items) - values.count("keep") - values.count("skip"),
        }

    def public_state(self):
        return {
            "manifest": self.manifest,
            "decisions": self.state["decisions"],
            "counts": self._counts(),
            "can_undo": bool(self.state["history"]),
        }

    def decide(self, item_id, decision, note=None):
        if item_id not in self.items:
            raise KeyError(item_id)
        if decision not in ("keep", "skip", None):
            raise ValueError(decision)
        with self.lock:
            previous = self.state["decisions"].get(item_id)
            self.state["history"].append({"id": item_id, "previous": previous})
            self.state["history"] = self.state["history"][-1000:]
            if decision is None:
                self.state["decisions"].pop(item_id, None)
            else:
                existing_note = "" if previous is None else previous.get("note", "")
                self.state["decisions"][item_id] = {
                    "decision": decision,
                    "note": existing_note if note is None else str(note),
                    "updated_at": now_iso(),
                }
            self.state["updated_at"] = now_iso()
            self._atomic_json(self.decisions_path, self.state)
            self._sync_exports()
            return self.public_state()

    def update_note(self, item_id, note):
        if item_id not in self.items:
            raise KeyError(item_id)
        with self.lock:
            current = dict(self.state["decisions"].get(item_id, {}))
            if not current:
                current = {"decision": None}
            current["note"] = str(note)
            current["updated_at"] = now_iso()
            self.state["decisions"][item_id] = current
            self.state["updated_at"] = now_iso()
            self._atomic_json(self.decisions_path, self.state)
            self._sync_exports()
            return self.public_state()

    def undo(self):
        with self.lock:
            if not self.state["history"]:
                return self.public_state()
            action = self.state["history"].pop()
            if action["previous"] is None:
                self.state["decisions"].pop(action["id"], None)
            else:
                self.state["decisions"][action["id"]] = action["previous"]
            self.state["updated_at"] = now_iso()
            self._atomic_json(self.decisions_path, self.state)
            self._sync_exports()
            return self.public_state()

    def _sync_exports(self):
        kept = []
        for item_id, decision in self.state["decisions"].items():
            if decision.get("decision") != "keep" or item_id not in self.items:
                continue
            kept.append({**self.items[item_id], "review": decision})
        kept.sort(key=lambda item: (item["dataset"], item["scene"], item["scene_rank"]))
        payload = {
            "version": 1,
            "updated_at": self.state.get("updated_at"),
            "count": len(kept),
            "items": kept,
        }
        self._atomic_json(self.selection_root / "kept_manifest.json", payload)
        fields = [
            "id", "dataset", "scene", "sample_id", "idx1", "idx2", "scene_rank",
            "score", "split_label", "note", "source_asset_dir", "geometry_asset_dir",
        ]
        temporary = self.selection_root / "kept.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in kept:
                writer.writerow({**item, "note": item["review"].get("note", "")})
        os.replace(temporary, self.selection_root / "kept.csv")

        kept_root = self.selection_root / "kept"
        kept_root.mkdir(parents=True, exist_ok=True)
        desired = set()
        for item in kept:
            destination = kept_root / item["dataset"] / safe_name(item["sample_id"])
            desired.add(destination.resolve(strict=False))
            destination.mkdir(parents=True, exist_ok=True)
            links = {
                "source_assets": Path(item["source_asset_dir"]),
                "geometry_assets": Path(item["geometry_asset_dir"]),
            }
            for name, target in links.items():
                link = destination / name
                if link.is_symlink() and link.resolve() == target.resolve():
                    continue
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target.resolve(), target_is_directory=True)
            self._atomic_json(destination / "selection.json", item)

        for dataset_dir in kept_root.iterdir() if kept_root.is_dir() else []:
            if not dataset_dir.is_dir():
                continue
            for candidate_dir in dataset_dir.iterdir():
                if candidate_dir.resolve(strict=False) not in desired:
                    shutil.rmtree(candidate_dir)
            if not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()


class ReviewHandler(SimpleHTTPRequestHandler):
    store = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.store.root), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def _send_json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.store.public_state())
            return
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/review_app/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/decision":
                result = self.store.decide(
                    str(payload["id"]), payload.get("decision"), payload.get("note")
                )
            elif path == "/api/note":
                result = self.store.update_note(str(payload["id"]), payload.get("note", ""))
            elif path == "/api/undo":
                result = self.store.undo()
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    args = parse_args()
    root = args.root.resolve()
    manifest = root / "review_app/review_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Build review assets first: {manifest}")
    ReviewHandler.store = ReviewStore(root)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{args.port}/review_app/",
                "root": str(root),
                "items": len(ReviewHandler.store.items),
                "selection": str(ReviewHandler.store.selection_root),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
