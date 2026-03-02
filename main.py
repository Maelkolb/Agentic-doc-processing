#!/usr/bin/env python3
"""
CLI entrypoint for the agentic document processor.
Usage: python main.py <image_path> [--no-gui]
"""

import argparse
import os
import sys

# Ensure src is on path when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_core.messages import HumanMessage

from agentic_doc.agent import build_agent
from agentic_doc.agent.callbacks import StreamingAgentCallback


def main():
    parser = argparse.ArgumentParser(description="Process a document image with the agentic pipeline.")
    parser.add_argument("image_path", help="Path to the document image file")
    parser.add_argument("--no-gui", action="store_true", help="Do not open GUI (e.g. when running headless)")
    parser.add_argument("--no-callbacks", action="store_true", help="Do not attach logging callbacks")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    agent_executor, state, logger = build_agent()
    if not args.no_callbacks:
        callback = StreamingAgentCallback(logger)
        config = {"configurable": {"callbacks": [callback]}}
    else:
        config = {}

    user_message = f"""Process this document image completely: {args.image_path}

Follow the full pipeline: assess, enhance if recommended, detect regions, detect lines, get transcription plan, transcribe every text region (use transcribe_with_llm for tables and images), compile transcription, then export to PageXML, Markdown, and HTML. Use the image path exactly as given for all tool calls."""

    print(f"Running agent on: {args.image_path}")
    print("---")
    result = agent_executor.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    print("---")
    if result and "messages" in result:
        last = result["messages"][-1]
        if hasattr(last, "content") and last.content:
            print("Final response:", last.content[:500], "..." if len(str(last.content)) > 500 else "")

    if not args.no_gui and getattr(state, "output_files", None):
        try:
            from agentic_doc.gui.panel import create_gui_panel_v12
            gui = create_gui_panel_v12()
            print("GUI panel created (display in Jupyter or call gui.display() if available).")
        except Exception as e:
            print(f"GUI not started: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
