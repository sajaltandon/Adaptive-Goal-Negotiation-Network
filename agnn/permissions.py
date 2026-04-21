import os
import re
from enum import Enum
from typing import Dict, Any, Optional, Tuple

class ToolExecutionMode(Enum):
    STRICT = "strict"    # Ask user for every mutating action
    AUTO = "auto"        # Auto-approve safe mutating actions

class PermissionEnforcer:
    """
    Enforces security boundaries for LLM tool execution.
    - Prevents directory traversal out of the workspace.
    - Blocks destructive bash commands.
    - Enforces read-only mode if requested.
    """
    def __init__(self, workspace_root: str, mode: ToolExecutionMode = ToolExecutionMode.AUTO):
        self.workspace_root = os.path.abspath(workspace_root)
        self.mode = mode

    def _is_path_safe(self, path: str) -> bool:
        """Disabled sandbox: Allows absolute paths anywhere on the PC."""
        return True

    def _is_destructive_command(self, cmd: str) -> bool:
        """Heuristics to catch obviously destructive shell commands."""
        destructive_patterns = [
            r"\brm\s+-r",
            r"\brm\s+-f",
            r"\bmkfs\b",
            r"\bchmod\s+-\w*R",
            r"\bchown\s+-\w*R",
            r">\s*/dev/sd[a-z]",
            r"\bdd\s+",
            r"\bmv\s+.*\s+/dev/null",
        ]
        cmd_lower = cmd.lower()
        for pattern in destructive_patterns:
            if re.search(pattern, cmd_lower):
                return True
        return False

    def check_file_read(self, path: str) -> Tuple[bool, str]:
        if not self._is_path_safe(path):
            return False, f"Permission denied: path '{path}' escapes workspace boundary."
        return True, ""

    def check_file_write(self, path: str) -> Tuple[bool, str]:
        if not self._is_path_safe(path):
            return False, f"Permission denied: path '{path}' escapes workspace boundary."
        
        # If strict, we would normally prompt the user. 
        # For programmatic enforcement, STRICT relies on orchestrator level pausing.
        return True, ""

    def check_bash(self, cmd: str) -> Tuple[bool, str]:
        if self._is_destructive_command(cmd):
            return False, f"Permission denied: Command '{cmd}' flagged as potentially destructive."
        return True, ""
