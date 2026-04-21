"""
AGNN Model Manager

Interacts with the local LLM server (LM Studio API) to dynamically query model capabilities, 
load/unload models gracefully, and determine context limits autonomously, replacing
hardcoded configuration strings.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from urllib.error import URLError
from typing import List, Dict, Any, Optional
import threading

class ModelManager:
    """Manages dynamic model loading, unloading, and capability discovery."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Caches
        self._all_models: List[Dict[str, Any]] = []
        self._loaded_models: List[Dict[str, Any]] = []
        self._context_lengths: Dict[str, int] = {}
        self._instance_ids: Dict[str, str] = {}
        self._lock = threading.Lock()

        # Session-level sticky blacklist: models that failed too many times
        # are permanently removed from rotation for this run.
        self._session_blacklist: set = set()
        self._failure_counts: Dict[str, int] = {}
        self.MAX_FAILURES_BEFORE_BLACKLIST = 3

        self.refresh()

    def refresh(self) -> None:
        """Fetch models from the LM Studio /api/v1/models endpoint."""
        with self._lock:
            self._do_refresh()
            
    def _do_refresh(self) -> None:
        url = f"{self.base_url}/api/v1/models"
        self._all_models = []
        self._loaded_models = []
        self._instance_ids = {}
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = data.get("models", data.get("data", []))
                    
                    for m in models:
                        model_id = m.get("id") or m.get("name") or m.get("key")
                        if not model_id:
                            continue
                            
                        self._all_models.append(m)
                        
                        # Check if loaded (LM Studio 0.2.x specific format)
                        instances = m.get("loaded_instances", [])
                        if instances:
                            self._loaded_models.append(m)
                            inst = instances[0] if isinstance(instances[0], dict) else {}
                            instance_id = inst.get("id") or inst.get("instance_id")
                            if instance_id:
                                self._instance_ids[model_id] = str(instance_id)
                            # Extract true context length dynamically
                            ctx_len = inst.get("config", {}).get("context_length")
                            if ctx_len:
                                self._context_lengths[model_id] = int(ctx_len)
                        
                        # Fallback to max_context_length if loaded_instances isn't found
                        max_ctx = m.get("max_context_length")
                        if max_ctx and model_id not in self._context_lengths:
                            self._context_lengths[model_id] = int(max_ctx)
        except Exception as e:
            print(f"[ModelManager] Failed to fetch models from {url}: {e}")
            # If standard OpenAI /v1/models is the only thing supported, fallback to it
            self._fallback_openai_models()

    def _fallback_openai_models(self) -> None:
        url = f"{self.base_url}/v1/models"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = data.get("data", [])
                    for m in models:
                        model_id = m.get("id")
                        if model_id:
                            self._all_models.append(m)
                            self._loaded_models.append(m) # Assume loaded
        except Exception:
            pass

    def get_loaded_model_ids(self) -> List[str]:
        """Return a list of currently running/loaded model IDs."""
        return [m.get("id") or m.get("key") for m in self._loaded_models]
        
    def get_unloaded_model_ids(self) -> List[str]:
        """Return a list of models available on disk but not in RAM."""
        loaded = set(self.get_loaded_model_ids())
        return [m.get("id") or m.get("key") for m in self._all_models if (m.get("id") or m.get("key")) not in loaded]

    def get_context_length(self, model_id: str, default: int = 4096) -> int:
        """Autonomously determine the context length of a model."""
        return self._context_lengths.get(model_id, default)

    def load_model(self, model_id: str) -> bool:
        """Attempt to load a model into memory."""
        if self._is_cloud_model(model_id):
            # Cloud models are not loadable via LM Studio model endpoints.
            return True
        with self._lock:
            print(f"[ModelManager] Attempting to load model '{model_id}'...")
            url = f"{self.base_url}/api/v1/models/load"
            payload = json.dumps({ "model": model_id }).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=60.0) as response:
                    if response.status in (200, 201):
                        print(f"[ModelManager] Successfully loaded {model_id}.")
                        self._do_refresh()
                        return True
            except Exception as e:
                # print(f"[ModelManager] Failed to load {model_id}: {e}")
                pass
            return False

    def unload_model(self, model_id: str) -> bool:
        """Attempt to unload a model from memory to free VRAM."""
        if self._is_cloud_model(model_id):
            # Cloud models are not unloadable via LM Studio model endpoints.
            return True
        with self._lock:
            print(f"[ModelManager] Attempting to unload model '{model_id}'...")
            url = f"{self.base_url}/api/v1/models/unload"
            payload_obj: Dict[str, Any] = {"model": model_id}
            instance_id = self._instance_ids.get(model_id)
            if instance_id:
                payload_obj["instance_id"] = instance_id
            payload = json.dumps(payload_obj).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    if response.status in (200, 204):
                        print(f"[ModelManager] Successfully unloaded {model_id}.")
                        self._do_refresh()
                        return True
            except Exception as e:
                # Fallback to older LM studio endpoint
                try:
                    # DELETE /v1/models/{model_id}
                    url = f"{self.base_url}/v1/models/{urllib.parse.quote(model_id, safe='')}"
                    req = urllib.request.Request(url, method="DELETE")
                    with urllib.request.urlopen(req, timeout=10.0) as response:
                        print(f"[ModelManager] Successfully unloaded {model_id} (fallback).")
                        self._do_refresh()
                        return True
                except Exception as e2:
                    # print(f"[ModelManager] Failed to unload {model_id}: {e2}")
                    pass
            return False

    @staticmethod
    def _is_cloud_model(model_id: str) -> bool:
        m = (model_id or "").lower()
        return m.startswith("groq/") or m.startswith("models/gemini") or "openai" in m
        
    def report_failure(self, model_id: str) -> bool:
        """Report a model inference failure. Returns True if model is now blacklisted."""
        with self._lock:
            self._failure_counts[model_id] = self._failure_counts.get(model_id, 0) + 1
            if self._failure_counts[model_id] >= self.MAX_FAILURES_BEFORE_BLACKLIST:
                if model_id not in self._session_blacklist:
                    self._session_blacklist.add(model_id)
                    print(f"[ModelManager] ⛔ '{model_id}' blacklisted for this session "
                          f"after {self._failure_counts[model_id]} failures.")
                return True
        return False

    def is_blacklisted(self, model_id: str) -> bool:
        """Check if a model has been session-blacklisted."""
        return model_id in self._session_blacklist

    def suggest_replacement_model(self, current_model_id: str) -> Optional[str]:
        """Automatically pick a replacement model from disk if one fails."""
        self.refresh()
        unloaded = self.get_unloaded_model_ids()
        loaded = self.get_loaded_model_ids()

        # Exclude the current model AND any blacklisted models from candidates
        def _is_eligible(m: str) -> bool:
            return (
                m != current_model_id
                and "embedding" not in m.lower()
                and m not in self._session_blacklist
            )

        # Try to find something on disk that isn't the current model
        candidates = [m for m in unloaded if _is_eligible(m)]
        if candidates:
            return sorted(candidates, reverse=True)[0]

        # Or try another loaded model
        loaded_candidates = [m for m in loaded if _is_eligible(m)]
        if loaded_candidates:
            return loaded_candidates[0]

        # Last resort: if everything is blacklisted, return any loaded non-current model
        fallback = [m for m in loaded if m != current_model_id and "embedding" not in m.lower()]
        if fallback:
            print(f"[ModelManager] ⚠ All alternates blacklisted — using {fallback[0]} as emergency fallback.")
            return fallback[0]

        return None
