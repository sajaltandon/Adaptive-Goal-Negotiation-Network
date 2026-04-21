import os
import platform
import re
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from .permissions import PermissionEnforcer

MAX_READ_SIZE = 100 * 1024  # 100KB
MAX_WRITE_SIZE = 500 * 1024 # 500KB

class Tool:
    name: str = ""
    description: str = ""
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the OpenAI-compatible JSON schema for this tool."""
        raise NotImplementedError
        
    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        """Execute the tool safely."""
        raise NotImplementedError

class FileReadTool(Tool):
    name = "read_file"
    description = "Read the contents of a file within the workspace."
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to read."
                        }
                    },
                    "required": ["path"]
                }
            }
        }
        
    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        path = kwargs.get("path")
        if not path:
            return "Error: path is required."
            
        allowed, msg = enforcer.check_file_read(path)
        if not allowed:
            return msg
            
        full_path = os.path.abspath(os.path.join(enforcer.workspace_root, path))
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist."
            
        if os.path.getsize(full_path) > MAX_READ_SIZE:
            return f"Error: File exceeds maximum read size of {MAX_READ_SIZE//1024}KB."
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"--- Contents of {path} ---\n{content}"
        except UnicodeDecodeError:
            return f"Error: '{path}' appears to be a binary file."
        except Exception as e:
            return f"Error reading file: {str(e)}"

class FileWriteTool(Tool):
    name = "write_file"
    description = "Write or overwrite a file within the workspace."
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file."
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }
        
    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        path = kwargs.get("path")
        content = kwargs.get("content", "")
        if not path:
            return "Error: path is required."
            
        allowed, msg = enforcer.check_file_write(path)
        if not allowed:
            return msg
            
        if len(content.encode('utf-8')) > MAX_WRITE_SIZE:
            return f"Error: Content exceeds maximum write size of {MAX_WRITE_SIZE//1024}KB."
            
        full_path = os.path.abspath(os.path.join(enforcer.workspace_root, path))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}."
        except Exception as e:
            return f"Error writing file: {str(e)}"

class BashTool(Tool):
    name = "bash"
    description = "Execute a bash/shell command."
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }
        
    def _run_command(self, cmd: str, cwd: str, timeout: int = 30) -> Tuple[int, str, str]:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()

    def _translate_unix_to_windows(self, cmd: str) -> Optional[str]:
        trimmed = (cmd or "").strip()
        if not trimmed:
            return None
        lower = trimmed.lower()

        # Fast command-level remaps for common autonomous task patterns.
        if lower.startswith("ls "):
            # "ls <path>" -> dir "<path>" /ad /b to list folders by name.
            arg = trimmed[3:].strip()
            if arg:
                return f'dir {arg} /ad /b'
            return "dir"
        if lower == "ls":
            return "dir"
        if lower == "pwd":
            return "cd"
        if lower.startswith("cat "):
            return "type " + trimmed[4:].strip()

        return None

    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        cmd = kwargs.get("command")
        if not cmd:
            return "Error: command is required."
            
        allowed, msg = enforcer.check_bash(cmd)
        if not allowed:
            return msg
            
        try:
            return_code, out, err = self._run_command(cmd, enforcer.workspace_root, timeout=30)

            # Windows self-heal pass for unix-style commands (ls/cat/pwd).
            translated_cmd = None
            if return_code != 0 and platform.system().lower().startswith("win"):
                translated_cmd = self._translate_unix_to_windows(cmd)
                if translated_cmd and translated_cmd != cmd:
                    return_code, out, err = self._run_command(translated_cmd, enforcer.workspace_root, timeout=30)

            output = f"EXIT_CODE: {return_code}\n"
            output += f"COMMAND: {cmd}\n"
            if translated_cmd:
                output += f"AUTO_RETRY_COMMAND: {translated_cmd}\n"
            if out:
                output += f"STDOUT:\n{out}\n"
            if err:
                output += f"STDERR:\n{err}\n"
            if not out and not err:
                output += "Command finished with no output.\n"
            return output

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"


class ListDirsTool(Tool):
    name = "list_dirs"
    description = "List subdirectories for a given path (OS-safe and deterministic)."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative directory path."},
                    },
                    "required": ["path"]
                }
            }
        }

    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        path = kwargs.get("path")
        if not path:
            return "Error: path is required."

        allowed, msg = enforcer.check_file_read(path)
        if not allowed:
            return msg

        full_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(enforcer.workspace_root, path))
        if not os.path.exists(full_path):
            return f"Error: Directory '{path}' does not exist."
        if not os.path.isdir(full_path):
            return f"Error: '{path}' is not a directory."

        try:
            names = sorted(
                [name for name in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, name))],
                key=lambda s: s.lower()
            )
            if not names:
                return f"Directories (0) in {full_path}:"
            return f"Directories ({len(names)}) in {full_path}:\n" + "\n".join(names)
        except Exception as e:
            return f"Error listing directories: {str(e)}"


class VerifyFileTool(Tool):
    name = "verify_file"
    description = "Verify file existence and minimum size."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative file path."},
                        "min_bytes": {"type": "integer", "description": "Minimum expected file size in bytes.", "default": 1}
                    },
                    "required": ["path"]
                }
            }
        }

    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        path = kwargs.get("path")
        min_bytes = int(kwargs.get("min_bytes", 1) or 1)
        if not path:
            return "Error: path is required."

        allowed, msg = enforcer.check_file_read(path)
        if not allowed:
            return msg

        full_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(enforcer.workspace_root, path))
        exists = os.path.exists(full_path)
        is_file = os.path.isfile(full_path)
        size = os.path.getsize(full_path) if exists and is_file else 0
        ok = exists and is_file and size >= min_bytes
        return (
            f"VERIFY path={full_path}\n"
            f"exists={exists}\n"
            f"is_file={is_file}\n"
            f"size_bytes={size}\n"
            f"min_bytes={min_bytes}\n"
            f"ok={ok}"
        )


class WriteTextFileTool(Tool):
    name = "write_text_file"
    description = "Write text to a file (absolute or relative path), with optional append."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative file path."},
                        "content": {"type": "string", "description": "Content to write."},
                        "append": {"type": "boolean", "description": "Append instead of overwrite.", "default": False}
                    },
                    "required": ["path", "content"]
                }
            }
        }

    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        path = kwargs.get("path")
        content = kwargs.get("content", "")
        append = bool(kwargs.get("append", False))
        if not path:
            return "Error: path is required."
        if len(content.encode("utf-8")) > MAX_WRITE_SIZE:
            return f"Error: Content exceeds maximum write size of {MAX_WRITE_SIZE//1024}KB."

        allowed, msg = enforcer.check_file_write(path)
        if not allowed:
            return msg

        full_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(enforcer.workspace_root, path))
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "a" if append else "w"
        try:
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            size = os.path.getsize(full_path)
            return f"WRITE_OK path={full_path} bytes={size} append={append}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class ConvertMarkdownToPdfTool(Tool):
    name = "convert_markdown_to_pdf"
    description = "Convert a Markdown file to a simple text-based PDF."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Path to source Markdown file (absolute or relative)."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to output PDF file (absolute or relative)."
                        }
                    },
                    "required": ["input_path", "output_path"]
                }
            }
        }

    @staticmethod
    def _to_abs(path: str, root: str) -> str:
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(root, path))

    @staticmethod
    def _pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    @staticmethod
    def _wrap_line(text: str, width: int = 95) -> List[str]:
        s = text.rstrip("\n")
        if not s:
            return [""]
        words = s.split()
        if not words:
            return [""]
        out: List[str] = []
        cur = words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) <= width:
                cur += " " + w
            else:
                out.append(cur)
                cur = w
        out.append(cur)
        return out

    def _build_pdf(self, lines: List[str]) -> bytes:
        """
        Build a basic multi-page PDF with Helvetica font.
        This is intentionally simple and dependency-free for reliability.
        """
        page_height = 792
        top_margin = 50
        left_margin = 40
        line_height = 14
        usable_lines = max(1, (page_height - (top_margin * 2)) // line_height)

        # Paginate lines
        pages: List[List[str]] = []
        for i in range(0, len(lines), usable_lines):
            pages.append(lines[i:i + usable_lines])
        if not pages:
            pages = [[""]]

        objects: List[bytes] = []

        # 1: Catalog, 2: Pages root, 3: Font
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [] /Count 0 >>")  # placeholder
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        page_obj_ids: List[int] = []
        content_obj_ids: List[int] = []

        for page_lines in pages:
            # Content stream
            content_ops = ["BT", f"/F1 11 Tf", f"{left_margin} {page_height - top_margin} Td"]
            first = True
            for line in page_lines:
                safe = self._pdf_escape(line)
                if first:
                    content_ops.append(f"({safe}) Tj")
                    first = False
                else:
                    content_ops.append(f"0 -{line_height} Td")
                    content_ops.append(f"({safe}) Tj")
            content_ops.append("ET")
            stream_data = "\n".join(content_ops).encode("latin-1", errors="replace")
            content_obj = (
                f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
                + stream_data
                + b"\nendstream"
            )
            objects.append(content_obj)
            content_obj_ids.append(len(objects))

            # Page object (parent=2, resources font=3)
            page_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)-0} 0 R >>"
            ).encode("latin-1")
            objects.append(page_obj)
            page_obj_ids.append(len(objects))

        # Fix /Pages object with kids + count
        kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        pages_obj = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>".encode("latin-1")
        objects[1] = pages_obj

        # Assemble full PDF
        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n")
        offsets = [0]  # xref object 0 placeholder

        for i, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("latin-1")
        )
        return bytes(pdf)

    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        input_path = kwargs.get("input_path")
        output_path = kwargs.get("output_path")
        if not input_path or not output_path:
            return "Error: input_path and output_path are required."

        allowed, msg = enforcer.check_file_read(input_path)
        if not allowed:
            return msg
        allowed, msg = enforcer.check_file_write(output_path)
        if not allowed:
            return msg

        src = self._to_abs(input_path, enforcer.workspace_root)
        dst = self._to_abs(output_path, enforcer.workspace_root)

        if not os.path.exists(src):
            return f"Error: input file '{input_path}' does not exist."
        if not os.path.isfile(src):
            return f"Error: input path '{input_path}' is not a file."

        try:
            with open(src, "r", encoding="utf-8") as f:
                md = f.read()
            if not md.strip():
                return "Error: input markdown file is empty."

            normalized_lines: List[str] = []
            for raw in md.splitlines():
                # Basic markdown stripping for PDF text readability
                line = raw
                line = re.sub(r"^#{1,6}\s*", "", line)
                line = re.sub(r"`([^`]+)`", r"\1", line)
                line = line.replace("**", "").replace("*", "")
                line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
                wrapped = self._wrap_line(line, width=95)
                normalized_lines.extend(wrapped)

            pdf_bytes = self._build_pdf(normalized_lines)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(pdf_bytes)

            size = os.path.getsize(dst)
            return f"PDF_OK input={src} output={dst} size_bytes={size}"
        except Exception as e:
            return f"Error converting markdown to PDF: {str(e)}"

import threading
import json
import uuid

class MCPClient:
    """Lightweight JSON-RPC over stdio client for Model Context Protocol"""
    def __init__(self, command: str, args: List[str]):
        import sys as _sys
        # On Windows, npx/node commands need shell=True to resolve .cmd extensions
        use_shell = _sys.platform == "win32"
        cmd = [command] + args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=use_shell,
        )
        self.lock = threading.Lock()
        self.responses = {}
        self.events = {}
        
        self.reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader_thread.start()
        
        init_resp = self.call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agnn", "version": "1.0.0"}
        })
        self.notify("notifications/initialized", {})

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                msg = json.loads(line)
                if "id" in msg:
                    self.responses[msg["id"]] = msg
                    if msg["id"] in self.events:
                        self.events[msg["id"]].set()
            except Exception:
                pass

    def call(self, method: str, params: dict) -> dict:
        req_id = str(uuid.uuid4())
        req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        
        evt = threading.Event()
        self.events[req_id] = evt
        
        with self.lock:
            if not self.process.stdin:
                raise Exception("Process stdin is closed")
            self.process.stdin.write(json.dumps(req) + "\n")
            self.process.stdin.flush()
            
        evt.wait(timeout=10)
        
        resp = self.responses.pop(req_id, None)
        self.events.pop(req_id, None)
        
        if not resp:
            raise Exception("MCP Timeout")
        if "error" in resp:
            raise Exception(resp["error"])
        return resp.get("result", {})

    def notify(self, method: str, params: dict):
        req = {"jsonrpc": "2.0", "method": method, "params": params}
        with self.lock:
            if self.process.stdin:
                self.process.stdin.write(json.dumps(req) + "\n")
                self.process.stdin.flush()
            
    def list_tools(self):
        resp = self.call("tools/list", {})
        return resp.get("tools", [])
        
    def call_tool(self, name: str, args: dict):
        resp = self.call("tools/call", {"name": name, "arguments": args})
        content = resp.get("content", [])
        if content and isinstance(content, list):
            # Combine text parts if there are multiple
            text_parts = []
            for c in content:
                if c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                else:
                    text_parts.append(str(c))
            return "\n".join(text_parts)
        return str(resp)

class DynamicMCPTool(Tool):
    def __init__(self, mcp_client: MCPClient, tool_info: Dict[str, Any]):
        self.mcp_client = mcp_client
        self.name = tool_info["name"]
        self.description = tool_info.get("description", "")
        self.input_schema = tool_info.get("inputSchema", {})
        
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }
        
    def execute(self, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        try:
            return self.mcp_client.call_tool(self.name, kwargs)
        except Exception as e:
            return f"MCP Error: {str(e)}"

class ToolRegistry:
    BUILTIN_TOOL_ORDER = (
        "list_dirs",
        "read_file",
        "write_file",
        "write_text_file",
        "verify_file",
        "convert_markdown_to_pdf",
        "bash",
    )

    def __init__(self):
        self.tools: Dict[str, Tool] = {
            "read_file": FileReadTool(),
            "write_file": FileWriteTool(),
            "write_text_file": WriteTextFileTool(),
            "list_dirs": ListDirsTool(),
            "verify_file": VerifyFileTool(),
            "convert_markdown_to_pdf": ConvertMarkdownToPdfTool(),
            "bash": BashTool(),
        }
        self.force_full_tool_schema = os.environ.get("AGNN_FORCE_FULL_TOOL_SCHEMA", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        # Failure-driven trimming stays enabled by default, but can still be
        # disabled explicitly for debugging if needed.
        self.enable_tool_degradation = os.environ.get("AGNN_ENABLE_TOOL_DEGRADATION", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        
    def load_mcp_server(self, command: str, args: List[str]):
        try:
            client = MCPClient(command, args)
            mcp_tools = client.list_tools()
            for t in mcp_tools:
                tool = DynamicMCPTool(client, t)
                self.tools[tool.name] = tool
            print(f"[Tools] Successfully loaded {len(mcp_tools)} tools from MCP server '{command}'.")
        except Exception as e:
            print(f"[Tools] Failed to load MCP server '{command}': {e}")
        
    @staticmethod
    def _contains_any(text: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _hint_tokens(task_hint: str) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "your",
            "their", "have", "there", "about", "when", "then", "than", "also",
            "using", "need", "want", "just", "make", "full", "many", "more",
            "into", "onto", "after", "before", "while", "within", "should",
            "could", "would", "them", "they", "what", "which", "where",
        }
        tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", (task_hint or "").lower())
        unique: List[str] = []
        seen = set()
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            unique.append(token)
        return unique

    def _score_builtin_tools(self, task_hint: str, phase_type: str) -> Dict[str, int]:
        text = f"{phase_type or ''}\n{task_hint or ''}".lower()
        scores = {name: 0 for name in self.BUILTIN_TOOL_ORDER}

        explore_patterns = [
            r"\banaly[sz]\w*\b", r"\bdebug\w*\b", r"\binspect\w*\b", r"\breview\w*\b",
            r"\btrace\w*\b", r"\bexplor\w*\b", r"\bscan\w*\b", r"\bbrowse\w*\b",
            r"\brepo\b", r"\brepositories?\b", r"\bworkspace\b", r"\bproject\b",
            r"\bfolder\b", r"\bdirector\w*\b", r"\bpath\b", r"\blocate\b", r"\bfind\b",
            r"\bissue\b", r"\berror\w*\b",
        ]
        read_patterns = explore_patterns + [
            r"\bread\b", r"\bfile\b", r"\bsource\b", r"\bfunction\b", r"\bclass\b",
            r"\bmodule\b", r"\bstack\b", r"\btraceback\b", r"\blog\b",
        ]
        code_write_patterns = [
            r"\bfix\w*\b", r"\bpatch\w*\b", r"\bedit\w*\b", r"\bmodif\w*\b",
            r"\bupdate\w*\b", r"\brefactor\w*\b", r"\bimplement\w*\b",
            r"\bwrite code\b", r"\bchange(?: the)? code\b",
            r"\bcreate (?:a |an )?(?:file|component|module|script|function|class)\b",
        ]
        text_write_patterns = [
            r"\bmarkdown\b", r"\breport\b", r"\bdocument\b", r"\bsummary\b",
            r"\bartifact\b", r"\bdeliverable\b", r"\bnotes?\b", r"\bproposal\b",
            r"\bplan\b", r"\bwrite[- ]?up\b",
        ]
        verify_patterns = [
            r"\bverify\w*\b", r"\bvalidat\w*\b", r"\bcheck\w*\b", r"\bconfirm\w*\b",
            r"\bensure\b", r"\bnon[- ]?empty\b", r"\bexists?\b",
        ]
        pdf_patterns = [r"\bpdf\b", r"\bexport\b", r"\bprintable\b"]
        shell_patterns = [
            r"\bbash\b", r"\bshell\b", r"\bpowershell\b", r"\bcmd\b", r"\bterminal\b",
            r"\bpytest\b", r"\bnpm\b", r"\bpnpm\b", r"\byarn\b", r"\bpip\b",
            r"\bgit\b", r"\bbuild\b", r"\binstall\w*\b", r"\bserver\b",
            r"\bstdout\b", r"\bstderr\b", r"\bport\b", r"\blogs?\b",
        ]

        if self._contains_any(text, explore_patterns):
            scores["list_dirs"] += 6
        if self._contains_any(text, read_patterns):
            scores["read_file"] += 7
        if self._contains_any(text, code_write_patterns):
            scores["write_file"] += 7
            scores["read_file"] += 3
        if self._contains_any(text, text_write_patterns):
            scores["write_text_file"] += 7
        if self._contains_any(text, verify_patterns):
            scores["verify_file"] += 5
        if self._contains_any(text, pdf_patterns):
            scores["convert_markdown_to_pdf"] += 8
            scores["write_text_file"] += 3
            scores["verify_file"] += 2
        if self._contains_any(text, shell_patterns):
            scores["bash"] += 7

        phase = (phase_type or "").lower().strip()
        if phase in {"research", "analysis", "review"}:
            scores["read_file"] += 3
            scores["list_dirs"] += 2
        if phase == "draft":
            scores["write_file"] += 1
            scores["write_text_file"] += 1
        if phase == "general" and max(scores.values()) == 0:
            scores["read_file"] += 4
            scores["list_dirs"] += 3

        if max(scores.values()) == 0:
            scores["read_file"] = 4
            scores["list_dirs"] = 3

        return scores

    def _score_dynamic_tools(self, task_hint: str) -> Dict[str, int]:
        hint_tokens = self._hint_tokens(task_hint)
        scores: Dict[str, int] = {}
        for name, tool in self.tools.items():
            if name in self.BUILTIN_TOOL_ORDER:
                continue
            haystack = f"{tool.name} {tool.description}".lower()
            name_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", tool.name.lower()))
            score = 0
            exact_name_match = False
            for token in hint_tokens:
                if token in name_tokens:
                    score += 3
                    exact_name_match = True
                elif token in haystack:
                    score += 1
            if exact_name_match and score >= 3:
                scores[name] = score
        return scores

    def _apply_failure_trim(self, tool_names: List[str], failure_level: int) -> List[str]:
        if not tool_names or not self.enable_tool_degradation or failure_level <= 0:
            return tool_names
        max_tools = 3 if failure_level == 1 else 2
        return tool_names[:max_tools]

    def _select_tool_names(self, task_hint: str, phase_type: str, failure_level: int) -> List[str]:
        builtin_scores = self._score_builtin_tools(task_hint, phase_type)
        dynamic_scores = self._score_dynamic_tools(task_hint)
        tie_order = {name: idx for idx, name in enumerate(self.tools.keys())}
        ranked = sorted(
            [item for item in {**builtin_scores, **dynamic_scores}.items() if item[1] > 0],
            key=lambda item: (-item[1], tie_order.get(item[0], 999)),
        )
        selected = [name for name, _ in ranked]
        return self._apply_failure_trim(selected, failure_level)

    def get_schemas(
        self,
        model: str = "",
        failure_level: int = 0,
        task_hint: str = "",
        phase_type: str = "",
    ) -> List[Dict[str, Any]]:
        schema_map = {name: tool.get_schema() for name, tool in self.tools.items()}
        if self.force_full_tool_schema:
            return [schema_map[name] for name in self.tools.keys()]

        selected_names = self._select_tool_names(task_hint=task_hint, phase_type=phase_type, failure_level=failure_level)
        if not selected_names:
            selected_names = ["read_file", "list_dirs"]

        return [schema_map[name] for name in selected_names if name in schema_map]
        
    def execute(self, tool_name: str, kwargs: Dict[str, Any], enforcer: PermissionEnforcer) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Unknown tool '{tool_name}'"
        return tool.execute(kwargs, enforcer)
