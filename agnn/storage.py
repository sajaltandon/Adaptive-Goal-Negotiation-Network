"""
Tier-3: Inter-Session Memory ("The Brain")

Persists agent relationships (IFCM) and learned role embeddings across sessions.
Allows the team to "remember" who works best together and specific agent strengths.
"""

import json
import os
from typing import Dict, Any, Optional

class AgntMemory:
    """
    Persistent memory storage for AGNN.
    Saves/Loads:
    1. IFCM (Information Flow Control Matrix) - Inter-agent influence
    2. Role Embeddings - Learned agent capabilities
    """
    
    def __init__(self, storage_dir: str = "agnn/data", filename: str = "brain.json"):
        self.storage_path = os.path.join(storage_dir, filename)
        self.data = {
            "ifcm": {},
            "embeddings": {},
            "negotiation_patterns": {},
            "session_count": 0,
            "last_updated": "",
            "policy_state": {}
        }
        self._ensure_dir()
        self.load()
        
    def _ensure_dir(self):
        """Ensure storage directory exists"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
    def load(self):
        """Load memory from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    stored_data = json.load(f)
                    self.data.update(stored_data)
                print(f"[Memory] Loaded brain from {self.storage_path} (Sessions: {self.data.get('session_count', 0)})")
            except Exception as e:
                print(f"[Memory] Load failed: {e}")
        else:
            print("[Memory] No existing brain found. Starting fresh.")
            
    def save(self):
        """Save memory to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            # print(f"[Memory] Saved brain to {self.storage_path}")
        except Exception as e:
            print(f"[Memory] Save failed: {e}")
            
    def update_session_stats(self):
        """Update session counter"""
        self.data["session_count"] = self.data.get("session_count", 0) + 1
        from datetime import datetime
        self.data["last_updated"] = datetime.now().isoformat()
        self.save()

    def update_ifcm(self, ifcm: Dict[str, Dict[str, float]]):
        """
        Update the influence matrix.
        We merge new data with old data using a decay factor to keep 'historical' memory.
        """
        if not ifcm:
            return
            
        current_ifcm = self.data.get("ifcm", {})
        
        # Merge logic: Average old and new, or just update?
        # Let's take the new values but respect historical connections if new ones are missing
        # Actually, Orchestrator IFCM evolves during the session. We should just save the final state.
        # But if we want *cross-session* learning, we should load it at start.
        
        self.data["ifcm"] = ifcm
        self.save()

    def update_embeddings(self, embeddings: Dict[str, Any]):
        """Update agent role embeddings"""
        serialized = {}
        for agent_id, emb in embeddings.items():
            # Handle both object and dict (if already serialized)
            if hasattr(emb, "to_dict"):
                serialized[agent_id] = emb.to_dict()
            else:
                serialized[agent_id] = emb
        
        self.data["embeddings"] = serialized
        self.save()
    
    def get_ifcm(self) -> Dict[str, Dict[str, float]]:
        """Get stored IFCM"""
        return self.data.get("ifcm", {})


    def update_negotiation_patterns(self, task_type: str, stats: Dict[str, Any]):
        """Update aggregated negotiation memory per task type."""
        if not task_type:
            return

        patterns = self.data.setdefault("negotiation_patterns", {})
        current = patterns.get(task_type, {
            "runs": 0,
            "avg_rounds": 0.0,
            "avg_consensus": 0.0,
            "avg_turn_quality": 0.0,
            "reopen_rate": 0.0
        })

        runs = int(current.get("runs", 0)) + 1

        def ema(old: float, new: float, n: int) -> float:
            old = float(old)
            new = float(new)
            return old + (new - old) / max(1, n)

        current["avg_rounds"] = round(ema(current.get("avg_rounds", 0.0), stats.get("rounds", 0.0), runs), 4)
        current["avg_consensus"] = round(ema(current.get("avg_consensus", 0.0), stats.get("consensus_strength", 0.0), runs), 4)
        current["avg_turn_quality"] = round(ema(current.get("avg_turn_quality", 0.0), stats.get("turn_quality", 0.0), runs), 4)
        current["reopen_rate"] = round(ema(current.get("reopen_rate", 0.0), 1.0 if stats.get("reopened") else 0.0, runs), 4)
        current["runs"] = runs

        patterns[task_type] = current
        self.data["negotiation_patterns"] = patterns
        self.save()

    def get_negotiation_patterns(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get negotiation memory for one task type or all."""
        patterns = self.data.get("negotiation_patterns", {})
        if task_type:
            return patterns.get(task_type, {})
        return patterns

    def get_embeddings(self) -> Dict[str, Dict[str, float]]:
        """Get stored embeddings"""
        return self.data.get("embeddings", {})


    def update_policy_state(self, policy_state: Dict[str, Any]):
        """Persist adaptive controller state."""
        self.data["policy_state"] = policy_state or {}
        self.save()

    def get_policy_state(self) -> Dict[str, Any]:
        return self.data.get("policy_state", {})

    def update_handoff_memory(self, role_from: str, role_to: str, hus_score: float, turns_taken: int = 0):
        """Update cross-session handoff quality memory (role-to-role handoff outcomes)."""
        handoff_memory = self.data.setdefault("handoff_memory", {})
        key = f"{role_from.lower()[:20]} -> {role_to.lower()[:20]}"

        current = handoff_memory.get(key, {"runs": 0, "avg_hus": 0.5, "avg_turns": 10.0})
        runs = int(current.get("runs", 0)) + 1

        def ema(old: float, new: float, n: int) -> float:
            return float(old) + (float(new) - float(old)) / max(1, n)

        current["avg_hus"] = round(ema(current.get("avg_hus", 0.5), hus_score, runs), 4)
        current["avg_turns"] = round(ema(current.get("avg_turns", 10.0), float(turns_taken), runs), 2)
        current["runs"] = runs

        handoff_memory[key] = current
        self.data["handoff_memory"] = handoff_memory
        self.save()

    def get_handoff_memory(self) -> Dict[str, Any]:
        """Get all cross-session handoff quality memory."""
        return self.data.get("handoff_memory", {})
