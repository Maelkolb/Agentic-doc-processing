"""Rich agent logger: LogLevel, AgentLogEntry, RichAgentLogger."""

import json
import queue
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class LogLevel(Enum):
    THOUGHT = "thought"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTION = "action"
    OBSERVATION = "observation"
    DECISION = "decision"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    FINAL = "final"
    PLAN = "plan"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    STREAM = "stream"


@dataclass
class AgentLogEntry:
    timestamp: str
    iteration: int
    level: LogLevel
    content: str
    phase: str = ""
    tool_name: str = ""
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "level": self.level.value,
            "content": self.content,
            "phase": self.phase,
            "tool_name": self.tool_name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class RichAgentLogger:
    """Enhanced logger with rich terminal formatting and streaming support."""

    COLORS = {
        LogLevel.THOUGHT: "\033[38;5;75m",
        LogLevel.REASONING: "\033[38;5;147m",
        LogLevel.PLANNING: "\033[38;5;213m",
        LogLevel.ACTION: "\033[38;5;77m",
        LogLevel.OBSERVATION: "\033[38;5;220m",
        LogLevel.DECISION: "\033[38;5;135m",
        LogLevel.ERROR: "\033[38;5;196m",
        LogLevel.WARNING: "\033[38;5;208m",
        LogLevel.INFO: "\033[38;5;81m",
        LogLevel.SUCCESS: "\033[38;5;46m",
        LogLevel.FINAL: "\033[38;5;46m",
        LogLevel.PLAN: "\033[38;5;141m",
        LogLevel.TOOL_START: "\033[38;5;228m",
        LogLevel.TOOL_END: "\033[38;5;228m",
        LogLevel.PHASE_START: "\033[38;5;51m",
        LogLevel.PHASE_END: "\033[38;5;51m",
        LogLevel.STREAM: "\033[38;5;250m",
    }
    ICONS = {
        LogLevel.THOUGHT: "💭",
        LogLevel.REASONING: "🧠",
        LogLevel.PLANNING: "📋",
        LogLevel.ACTION: "⚡",
        LogLevel.OBSERVATION: "👁️",
        LogLevel.DECISION: "🎯",
        LogLevel.ERROR: "❌",
        LogLevel.WARNING: "⚠️",
        LogLevel.INFO: "ℹ️",
        LogLevel.SUCCESS: "✅",
        LogLevel.FINAL: "🏁",
        LogLevel.PLAN: "📝",
        LogLevel.TOOL_START: "▶️",
        LogLevel.TOOL_END: "⏹️",
        LogLevel.PHASE_START: "🚀",
        LogLevel.PHASE_END: "✓",
        LogLevel.STREAM: "···",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BOX_TOP = "╭" + "─" * 78 + "╮"
    BOX_BOTTOM = "╰" + "─" * 78 + "╯"
    BOX_SIDE = "│"
    BOX_DIVIDER = "├" + "─" * 78 + "┤"

    def __init__(self, verbose: bool = True):
        self.entries: List[AgentLogEntry] = []
        self.verbose = verbose
        self.current_iteration = 0
        self.current_phase = "initialization"
        self.phase_start_times: Dict[str, datetime] = {}
        self._event_queue: queue.Queue = queue.Queue()

    def log(self, level: LogLevel, content: str, **kwargs) -> AgentLogEntry:
        entry = AgentLogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            iteration=self.current_iteration,
            level=level,
            content=content,
            phase=self.current_phase,
            tool_name=kwargs.get("tool_name", ""),
            duration_ms=kwargs.get("duration_ms", 0),
            metadata=kwargs.get("metadata", {}),
        )
        self.entries.append(entry)
        self._event_queue.put(entry)
        if self.verbose:
            self._print_entry(entry)
        return entry

    def _print_entry(self, entry: AgentLogEntry) -> None:
        icon = self.ICONS.get(entry.level, "•")
        color = self.COLORS.get(entry.level, "")
        if entry.level == LogLevel.THOUGHT:
            self._print_boxed_thought(entry, icon, color)
        elif entry.level == LogLevel.REASONING:
            print(f"\n{color}{icon} [{entry.timestamp}] {self.BOLD}REASONING:{self.RESET}")
            for line in entry.content.split("\n")[:8]:
                print(f"{color}   {line}{self.RESET}")
        elif entry.level == LogLevel.PHASE_START:
            phase_name = entry.metadata.get("phase_name", entry.content)
            print(f"\n{color}{'═' * 80}{self.RESET}")
            print(f"{color}{icon}  PHASE: {self.BOLD}{phase_name.upper()}{self.RESET}")
            print(f"{color}{'═' * 80}{self.RESET}")
        elif entry.level == LogLevel.PHASE_END:
            duration = entry.metadata.get("duration", "")
            duration_str = f" ({duration})" if duration else ""
            print(f"\n{color}{icon} Phase complete: {entry.content}{duration_str}{self.RESET}")
        elif entry.level == LogLevel.ACTION:
            print(f"\n{color}{icon} [{entry.timestamp}] {self.BOLD}ACTION:{self.RESET} {entry.content}")
            if entry.metadata.get("args"):
                args_str = json.dumps(entry.metadata["args"], indent=2, default=str)
                for line in args_str.split("\n")[:10]:
                    print(f"{self.DIM}   {line}{self.RESET}")
        elif entry.level == LogLevel.OBSERVATION:
            print(f"\n{color}{icon} [{entry.timestamp}] OBSERVATION:{self.RESET}")
            preview = entry.content[:600] + ("..." if len(entry.content) > 600 else "")
            for line in preview.split("\n"):
                print(f"{self.DIM}   {line}{self.RESET}")
        elif entry.level == LogLevel.TOOL_START:
            print(f"\n{color}{icon} [{entry.timestamp}] Starting: {self.BOLD}{entry.tool_name or entry.content}{self.RESET}")
        elif entry.level == LogLevel.TOOL_END:
            duration = f"{entry.duration_ms:.0f}ms" if entry.duration_ms else ""
            print(f"{color}{icon} [{entry.timestamp}] Completed: {entry.tool_name or entry.content} {self.DIM}({duration}){self.RESET}")
        elif entry.level == LogLevel.ERROR:
            print(f"\n{color}{self.BOLD}{icon} ERROR: {entry.content}{self.RESET}")
        else:
            print(f"\n{color}{icon} [{entry.timestamp}] {entry.level.value.upper()}: {entry.content}{self.RESET}")

    def _print_boxed_thought(self, entry: AgentLogEntry, icon: str, color: str) -> None:
        print(f"\n{color}{self.BOX_TOP}{self.RESET}")
        header = f" {icon} THOUGHT (Step {entry.iteration + 1})"
        print(f"{color}{self.BOX_SIDE}{header:<78}{self.BOX_SIDE}{self.RESET}")
        print(f"{color}{self.BOX_DIVIDER}{self.RESET}")
        for line in entry.content.split("\n")[:15]:
            display_line = line[:76]
            padding = 76 - len(display_line)
            print(f"{color}{self.BOX_SIDE} {display_line}{' ' * padding}{self.BOX_SIDE}{self.RESET}")
        if len(entry.content.split("\n")) > 15:
            print(f"{color}{self.BOX_SIDE} {'... [truncated]':<76}{self.BOX_SIDE}{self.RESET}")
        print(f"{color}{self.BOX_BOTTOM}{self.RESET}")

    def thought(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.THOUGHT, content, **kwargs)

    def reasoning(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.REASONING, content, **kwargs)

    def action(self, tool_name: str, args: Dict = None) -> AgentLogEntry:
        return self.log(LogLevel.ACTION, f"Calling {tool_name}()", tool_name=tool_name, metadata={"args": args or {}})

    def observation(self, result: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.OBSERVATION, result, **kwargs)

    def phase_start(self, phase_name: str) -> AgentLogEntry:
        self.current_phase = phase_name
        self.phase_start_times[phase_name] = datetime.now()
        return self.log(LogLevel.PHASE_START, phase_name, metadata={"phase_name": phase_name})

    def phase_end(self, phase_name: str = None) -> AgentLogEntry:
        phase = phase_name or self.current_phase
        start = self.phase_start_times.get(phase)
        duration = f"{(datetime.now() - start).total_seconds():.2f}s" if start else ""
        return self.log(LogLevel.PHASE_END, phase, metadata={"duration": duration})

    def tool_start(self, tool_name: str) -> datetime:
        self.log(LogLevel.TOOL_START, tool_name, tool_name=tool_name)
        return datetime.now()

    def tool_end(self, tool_name: str, start_time: datetime) -> AgentLogEntry:
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return self.log(LogLevel.TOOL_END, tool_name, tool_name=tool_name, duration_ms=duration_ms)

    def info(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.INFO, content, **kwargs)

    def success(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.SUCCESS, content, **kwargs)

    def error(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.ERROR, content, **kwargs)

    def warning(self, content: str, **kwargs) -> AgentLogEntry:
        return self.log(LogLevel.WARNING, content, **kwargs)

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def next_iteration(self) -> None:
        self.current_iteration += 1

    def clear(self) -> None:
        self.entries = []
        self.current_iteration = 0
        self.current_phase = "initialization"
        self.phase_start_times = {}

    def get_full_trace(self) -> List[Dict]:
        return [e.to_dict() for e in self.entries]

    def get_summary(self) -> Dict:
        counts: Dict[str, int] = {}
        phases = set()
        for e in self.entries:
            counts[e.level.value] = counts.get(e.level.value, 0) + 1
            if e.phase:
                phases.add(e.phase)
        return {
            "total_entries": len(self.entries),
            "iterations": self.current_iteration + 1,
            "phases": list(phases),
            "counts_by_type": counts,
        }

    def export_json(self, filepath: str = None) -> str:
        data = json.dumps(self.get_full_trace(), indent=2)
        if filepath:
            with open(filepath, "w") as f:
                f.write(data)
        return data
