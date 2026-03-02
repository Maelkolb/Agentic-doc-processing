#!/usr/bin/env python3
"""
CLI entrypoint for the agentic document processor.
Usage: python main.py <image_path> [--no-gui]
With GUI (default): in Jupyter/Colab the panel is shown and updated live; from CLI the pipeline runs and GUI updates no-op unless in an IPython context.
"""

import argparse
import os
import sys

# Ensure src is on path when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentic_doc.agent import build_agent


def main():
    parser = argparse.ArgumentParser(description="Process a document image with the agentic pipeline.")
    parser.add_argument("image_path", help="Path to the document image file")
    parser.add_argument("--no-gui", action="store_true", help="Do not use GUI (run with invoke only, no panel)")
    parser.add_argument("--no-callbacks", action="store_true", help="Do not attach logging callbacks")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    agent_executor, state, logger = build_agent()

    user_message = (
        f"Process this document image completely: {args.image_path}\n\n"
        "Follow the full pipeline: assess, enhance if recommended, detect regions, detect lines, "
        "get transcription plan, transcribe every text region (use transcribe_with_llm for tables and images), "
        "compile transcription, then export to PageXML, Markdown, and HTML. Use the image path exactly as given for all tool calls."
    )

    if args.no_gui:
        from langchain_core.messages import HumanMessage
        if not args.no_callbacks:
            from agentic_doc.agent.callbacks import StreamingAgentCallback
            config = {"configurable": {"callbacks": [StreamingAgentCallback(logger)]}}
        else:
            config = {}
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
    else:
        from agentic_doc.gui.panel import run_with_gui
        try:
            result = run_with_gui(
                agent_executor,
                state,
                logger,
                args.image_path,
                task_prompt=user_message,
            )
            if not result and getattr(state, "output_files", None):
                print("Pipeline finished. View outputs in the GUI panel (Jupyter/Colab).")
        except Exception as e:
            print(f"GUI run failed: {e}", file=sys.stderr)
            raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
