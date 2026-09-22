#!/usr/bin/env python3
"""
Serena + Local LLM Agent Bridge (configurable version)
Reads settings from agent_config.yaml in the project directory.
Works with any llama-server model that supports function calling.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def load_config(config_path: str) -> dict:
    """Load YAML config, falling back to defaults."""
    defaults = {
        "llm": {
            "base_url": "http://127.0.0.1:8091/v1",
            "model": "gemma-4-e4b-qat",
            "max_tokens": 17000,
            "temperature": 0.6,
            "top_p": 0.95,
            "timeout": 4000.0,
        },
        "agent": {"max_turns": 30, "max_tool_result_chars": 8000},
        "serena": {"bin": "/home/walker/.local/bin/serena", "context": "", "project": ""},
        "output": {"filename": "index.html", "log_file": "agent_log.json"},
    }
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge user config over defaults
        for section, values in user_cfg.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
            else:
                defaults[section] = values
    return defaults


def build_system_prompt(config: dict, project_dir: str) -> str:
    """Build system prompt from system_prompt.txt + agent instructions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(script_dir, "system_prompt.txt")
    # Also check project dir
    if not os.path.exists(prompt_file):
        prompt_file = os.path.join(project_dir, "system_prompt.txt")

    custom_prompt = ""
    if os.path.exists(prompt_file):
        with open(prompt_file) as f:
            custom_prompt = f.read()

    agent_instructions = f"""\
You are a coding agent with access to Serena MCP tools for semantic code analysis and editing.

You are working on a project at: {project_dir}

CRITICAL INSTRUCTIONS:
- You MUST use the create_text_file tool to write files. Do NOT output code as text.
- The create_text_file tool takes two arguments: relative_path (e.g. "{config['output']['filename']}") and content (the full file text).
- When asked to create a file, call create_text_file with the complete file content.
- After writing, use read_file to verify the file was created correctly.
- Think step by step about what tools to call.

Available Serena tools will be provided as function definitions. Call them by name.
"""
    if custom_prompt:
        return custom_prompt + "\n\n" + agent_instructions
    return agent_instructions


def log_event(event: dict, log_file: str):
    event["timestamp"] = time.time()
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[{event.get('type','?')}] {event.get('summary','')}", flush=True)


async def get_serena_tools(session: ClientSession) -> list:
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        if "jet_brains" in tool.name or tool.name == "open_dashboard":
            continue
        schema = tool.input_schema if tool.input_schema else {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or "")[:800],
                "parameters": schema,
            }
        })
    return tools


async def call_serena_tool(session: ClientSession, tool_name: str, arguments: dict) -> str:
    result = await session.call_tool(tool_name, arguments=arguments)
    parts = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts) if parts else "(no output)"


def call_llm(messages: list, tools: list, cfg: dict) -> dict:
    payload = {
        "model": cfg["llm"]["model"],
        "messages": messages,
        "max_tokens": cfg["llm"]["max_tokens"],
        "temperature": cfg["llm"]["temperature"],
        "top_p": cfg["llm"]["top_p"],
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    with httpx.Client(timeout=cfg["llm"]["timeout"]) as client:
        resp = client.post(
            f"{cfg['llm']['base_url']}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def run_agent(task_prompt: str, cfg: dict, project_dir: str, log_file: str):
    serena_args = ["start-mcp-server", "--project", cfg["serena"].get("project") or project_dir]
    if cfg["serena"].get("context"):
        serena_args += ["--context", cfg["serena"]["context"]]

    server_params = StdioServerParameters(
        command=cfg["serena"]["bin"],
        args=serena_args,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ["HOME"],
            "USER": os.environ.get("USER", "walker"),
        },
    )

    system_prompt = build_system_prompt(cfg, project_dir)
    max_truncate = cfg["agent"]["max_tool_result_chars"]
    output_filename = cfg["output"]["filename"]

    log_event({"type": "start", "summary": f"Agent starting: {task_prompt[:80]}..."}, log_file)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=None) as session:
            await session.initialize()
            tools = await get_serena_tools(session)
            tool_names = [t["function"]["name"] for t in tools]
            log_event({"type": "tools", "summary": f"Discovered {len(tools)} Serena tools", "tools": tool_names}, log_file)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ]

            total_tokens = 0
            total_tool_calls = 0
            start_time = time.time()

            for turn in range(1, cfg["agent"]["max_turns"] + 1):
                log_event({"type": "llm_call", "summary": f"Turn {turn}: calling LLM ({len(messages)} msgs)", "turn": turn}, log_file)

                try:
                    response = call_llm(messages, tools, cfg)
                except Exception as e:
                    log_event({"type": "error", "summary": f"LLM call failed: {e}", "turn": turn}, log_file)
                    break

                choice = response["choices"][0]
                msg = choice["message"]
                usage = response.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)
                finish_reason = choice["finish_reason"]

                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    log_event({"type": "reasoning", "summary": f"Turn {turn} reasoning: {reasoning[:200]}...", "turn": turn, "reasoning_len": len(reasoning)}, log_file)

                tool_calls = msg.get("tool_calls", [])
                content = msg.get("content", "")

                if content:
                    log_event({"type": "response", "summary": f"Turn {turn}: {content[:200]}", "turn": turn, "content": content}, log_file)

                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {"id": tc["id"], "type": "function",
                             "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                            for tc in tool_calls
                        ],
                    })

                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}

                        log_event({"type": "tool_call", "summary": f"Turn {turn}: {tool_name}({json.dumps(args)[:150]})", "turn": turn, "tool": tool_name, "args": args}, log_file)
                        total_tool_calls += 1

                        try:
                            result_text = await call_serena_tool(session, tool_name, args)
                            if len(result_text) > max_truncate:
                                half = max_truncate // 2
                                result_text = result_text[:half] + f"\n\n... [truncated, {len(result_text)} chars total] ...\n\n" + result_text[-half//2:]
                            log_event({"type": "tool_result", "summary": f"Turn {turn}: {tool_name} returned {len(result_text)} chars", "turn": turn, "tool": tool_name, "result_len": len(result_text), "result_preview": result_text[:300]}, log_file)
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_text})
                        except Exception as e:
                            log_event({"type": "tool_error", "summary": f"Turn {turn}: {tool_name} failed: {e}", "turn": turn, "tool": tool_name, "error": str(e)}, log_file)
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"Error: {e}"})

                elif finish_reason == "stop":
                    # Fallback: extract and save if content contains code
                    if content and ("<!DOCTYPE" in content or "```html" in content):
                        if "```html" in content:
                            html_start = content.index("```html") + 7
                            html_content = content[html_start:content.rindex("```")].strip()
                        elif "<!DOCTYPE" in content:
                            html_content = content[content.index("<!DOCTYPE"):].strip()
                        else:
                            html_content = content

                        try:
                            result_text = await call_serena_tool(session, "create_text_file", {
                                "relative_path": output_filename, "content": html_content
                            })
                            log_event({"type": "fallback_save", "summary": f"Saved from content via create_text_file: {len(html_content)} chars", "turn": turn, "result": result_text}, log_file)
                        except Exception:
                            with open(os.path.join(project_dir, output_filename), "w") as f:
                                f.write(html_content)
                            log_event({"type": "fallback_save", "summary": f"Saved from content directly: {len(html_content)} chars", "turn": turn}, log_file)

                    log_event({"type": "complete", "summary": f"Agent finished at turn {turn}. {content[:300]}", "turn": turn, "content": content}, log_file)
                    break

                elif finish_reason == "length":
                    log_event({"type": "warning", "summary": f"Turn {turn}: hit max_tokens ({cfg['llm']['max_tokens']})", "turn": turn}, log_file)
                    messages.append({"role": "assistant", "content": content or "(continued)"})
                else:
                    log_event({"type": "warning", "summary": f"Turn {turn}: finish_reason={finish_reason}", "turn": turn}, log_file)
                    if content:
                        messages.append({"role": "assistant", "content": content})
                    else:
                        break

            elapsed = time.time() - start_time
            log_event({
                "type": "final",
                "summary": f"Done: {total_tool_calls} tool calls, {total_tokens} tokens, {elapsed:.1f}s",
                "total_turns": turn, "total_tool_calls": total_tool_calls,
                "total_tokens": total_tokens, "elapsed_seconds": elapsed,
                "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
            }, log_file)
            return total_tool_calls, total_tokens, elapsed


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, "agent_config.yaml")

    cfg = load_config(config_path)
    log_file = os.path.join(project_dir, cfg["output"]["log_file"])

    # Load task prompt
    prompt_file = os.path.join(project_dir, "game_prompt.txt")
    if len(sys.argv) > 1:
        task = sys.argv[1]
    elif os.path.exists(prompt_file):
        with open(prompt_file) as f:
            task = f.read()
    else:
        task = "Create a complete HTML5 game. Output the full HTML file."

    print(f"=== Serena + Local LLM Agent Bridge ===")
    print(f"  Model:    {cfg['llm']['model']} at {cfg['llm']['base_url']}")
    print(f"  Project:  {project_dir}")
    print(f"  Tokens:   {cfg['llm']['max_tokens']} per turn")
    print(f"  Timeout:  {cfg['llm']['timeout']}s")
    print(f"  Turns:    {cfg['agent']['max_turns']}")
    print(f"  Task:     {task[:80]}...")
    print(f"  Log:      {log_file}")
    print("=" * 60)

    with open(log_file, "w") as f:
        pass

    result = asyncio.run(run_agent(task, cfg, project_dir, log_file))
    print(f"\n=== RESULT: {result[0]} tool calls, {result[1]:,} tokens, {result[2]:.1f}s ===")