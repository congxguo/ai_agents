"""
Agent Harness PoC
==================
A minimal but complete example of an agent architecture:

  Harness  -> the Agent class that owns state, tools, hooks, and the LLM client
  Loop     -> Agent.run(): think -> (maybe call tools) -> observe -> repeat
  Hooks    -> named interception points other code can subscribe to,
              which can inspect, mutate, or abort the loop
  Tools    -> a small registry of callable functions the model can invoke

Run it directly:  python agent_harness.py

No API key is required. LLMClient has two implementations:
  - ScriptedMockLLM   -> deterministic stand-in, used by default so this
                         file runs end-to-end with no external dependency
  - AnthropicLLMClient -> real implementation using the Anthropic Messages
                         API + tool use, activated automatically if
                         ANTHROPIC_API_KEY is set in the environment
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================
# 1. Core data types
# ============================================================

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: str                                  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ModelResponse:
    """What the LLM layer hands back to the harness each turn."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"               # "end_turn" | "tool_use"
    usage: dict = field(default_factory=dict)


# ============================================================
# 2. Hook system
# ============================================================

class HookEvent(str, Enum):
    ON_START = "on_start"
    ON_STEP = "on_step"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_ERROR = "on_error"
    ON_FINISH = "on_finish"


class HookManager:
    """
    Registry of callbacks keyed by HookEvent.

    A hook receives the current context as kwargs and may:
      - return None                -> no effect
      - return a dict               -> merged into context (can override values,
                                        e.g. rewrite a tool call's arguments)
      - set ctx['abort'] = True     -> harness stops the loop after this event
    """

    def __init__(self):
        self._hooks: dict[HookEvent, list[Callable]] = {e: [] for e in HookEvent}

    def register(self, event: HookEvent, fn: Callable) -> Callable:
        self._hooks[event].append(fn)
        return fn

    def on(self, event: HookEvent):
        """Decorator form: @hooks.on(HookEvent.BEFORE_TOOL_CALL)"""
        def wrapper(fn):
            self.register(event, fn)
            return fn
        return wrapper

    def fire(self, event: HookEvent, **ctx) -> dict:
        for fn in self._hooks[event]:
            result = fn(**ctx)
            if isinstance(result, dict):
                ctx.update(result)
        return ctx


# ============================================================
# 3. Tools
# ============================================================

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                 # JSON-schema-ish, for the model's tool spec
    fn: Callable[..., str]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict):
        def wrapper(fn: Callable[..., str]):
            self._tools[name] = Tool(name, description, parameters, fn)
            return fn
        return wrapper

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        """Tool specs in Anthropic API's expected shape."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)
        try:
            output = tool.fn(**call.arguments)
            return ToolResult(call.id, call.name, str(output))
        except Exception as e:
            return ToolResult(call.id, call.name, f"{type(e).__name__}: {e}", is_error=True)


tools = ToolRegistry()


@tools.register(
    name="calculator",
    description="Evaluate a basic arithmetic expression, e.g. '23*19+4'.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("expression contains disallowed characters")
    return str(eval(expression, {"__builtins__": {}}, {}))


@tools.register(
    name="get_time",
    description="Get the current server time.",
    parameters={"type": "object", "properties": {}},
)
def get_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@tools.register(
    name="fun_fact",
    description="Get a fun fact about a given integer.",
    parameters={
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    },
)
def fun_fact(n: int) -> str:
    if n % 2 == 0:
        return f"{n} is an even number."
    return f"{n} is an odd number, and odd numbers can't be split evenly in two."


# ============================================================
# 4. LLM client layer (pluggable)
# ============================================================

class LLMClient:
    def call(self, messages: list[Message], system: str, tool_specs: list[dict]) -> ModelResponse:
        raise NotImplementedError


class ScriptedMockLLM(LLMClient):
    """
    Deterministic stand-in for a real model, purely so this file is
    runnable with zero setup. It inspects the conversation and decides
    what a reasonable tool-using model would do at each step.
    """

    def call(self, messages: list[Message], system: str, tool_specs: list[dict]) -> ModelResponse:
        tool_results_so_far = [m for m in messages if m.role == "tool"]
        have = {m.name for m in tool_results_so_far}

        if "calculator" not in have:
            return ModelResponse(
                text="I'll compute that first.",
                tool_calls=[ToolCall(id=str(uuid.uuid4()), name="calculator",
                                      arguments={"expression": "23*19+4"})],
                stop_reason="tool_use",
            )
        if "get_time" not in have:
            return ModelResponse(
                text="Now let me check the time too.",
                tool_calls=[ToolCall(id=str(uuid.uuid4()), name="get_time", arguments={})],
                stop_reason="tool_use",
            )
        if "fun_fact" not in have:
            calc_value = next(m.content for m in tool_results_so_far if m.name == "calculator")
            return ModelResponse(
                text="Let me grab a fun fact about that number.",
                tool_calls=[ToolCall(id=str(uuid.uuid4()), name="fun_fact",
                                      arguments={"n": int(calc_value)})],
                stop_reason="tool_use",
            )

        calc_value = next(m.content for m in tool_results_so_far if m.name == "calculator")
        time_value = next(m.content for m in tool_results_so_far if m.name == "get_time")
        fact_value = next(m.content for m in tool_results_so_far if m.name == "fun_fact")
        return ModelResponse(
            text=(f"23*19+4 = {calc_value}. The current server time is {time_value}. "
                  f"Fun fact: {fact_value}"),
            stop_reason="end_turn",
        )


class AnthropicLLMClient(LLMClient):
    """
    Real implementation using the Anthropic Messages API with tool use.
    Requires ANTHROPIC_API_KEY in the environment and the `anthropic`
    package installed (`pip install anthropic`).
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # imported lazily so the mock path has no dependency
        self.client = anthropic.Anthropic()
        self.model = model

    def call(self, messages: list[Message], system: str, tool_specs: list[dict]) -> ModelResponse:
        api_messages = []
        for m in messages:
            if m.role == "user":
                api_messages.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                api_messages.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=api_messages,
            tools=tool_specs,
        )

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=b.input)
                 for b in resp.content if b.type == "tool_use"]
        return ModelResponse(
            text=text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=dict(resp.usage) if resp.usage else {},
        )


# ============================================================
# 5. The harness: Agent class + main loop
# ============================================================

class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        hooks: HookManager,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 8,
        max_history: int = 40,
    ):
        self.llm = llm
        self.tools = tools
        self.hooks = hooks
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_history = max_history
        self.messages: list[Message] = []

    def _trim_history(self):
        # simple sliding-window memory management
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def run(self, user_input: str) -> str:
        ctx = self.hooks.fire(HookEvent.ON_START, agent=self, user_input=user_input)
        if ctx.get("abort"):
            return "[aborted before start]"

        self.messages.append(Message(role="user", content=user_input))

        for step in range(self.max_steps):
            ctx = self.hooks.fire(HookEvent.ON_STEP, agent=self, step=step)
            if ctx.get("abort"):
                return "[aborted mid-loop]"

            try:
                ctx = self.hooks.fire(
                    HookEvent.BEFORE_LLM_CALL, agent=self, step=step, messages=self.messages
                )
                if ctx.get("abort"):
                    return "[aborted before model call]"

                response = self.llm.call(self.messages, self.system_prompt, self.tools.specs())

                ctx = self.hooks.fire(
                    HookEvent.AFTER_LLM_CALL, agent=self, step=step, response=response
                )
                if ctx.get("abort"):
                    return "[aborted after model call]"

            except Exception as e:
                self.hooks.fire(HookEvent.ON_ERROR, agent=self, step=step,
                                 error=e, traceback=traceback.format_exc())
                return f"[error during model call: {e}]"

            self.messages.append(
                Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )

            if response.stop_reason != "tool_use" or not response.tool_calls:
                self._trim_history()
                self.hooks.fire(HookEvent.ON_FINISH, agent=self, final_text=response.text)
                return response.text

            for call in response.tool_calls:
                ctx = self.hooks.fire(
                    HookEvent.BEFORE_TOOL_CALL, agent=self, step=step, tool_call=call
                )
                if ctx.get("abort"):
                    return f"[aborted before tool call: {call.name}]"
                call = ctx.get("tool_call", call)   # hooks may rewrite the call

                result = self.tools.execute(call)

                ctx = self.hooks.fire(
                    HookEvent.AFTER_TOOL_CALL, agent=self, step=step,
                    tool_call=call, result=result
                )
                if ctx.get("abort"):
                    return f"[aborted after tool call: {call.name}]"
                result = ctx.get("result", result)

                self.messages.append(
                    Message(role="tool", content=result.content,
                            tool_call_id=result.tool_call_id, name=result.name)
                )

            self._trim_history()

        self.hooks.fire(HookEvent.ON_ERROR, agent=self, error="max_steps exceeded")
        return "[stopped: max_steps exceeded]"


# ============================================================
# 6. Example hooks: logging, safety guard, metrics
# ============================================================

def build_default_hooks() -> HookManager:
    hooks = HookManager()
    metrics = {"llm_calls": 0, "tool_calls": 0, "start_time": None}

    @hooks.on(HookEvent.ON_START)
    def log_start(**ctx):
        metrics["start_time"] = time.time()
        print(f"[hook:on_start] user_input={ctx['user_input']!r}")

    @hooks.on(HookEvent.BEFORE_LLM_CALL)
    def log_before_llm(**ctx):
        metrics["llm_calls"] += 1
        print(f"[hook:before_llm_call] step={ctx['step']} history_len={len(ctx['messages'])}")

    @hooks.on(HookEvent.AFTER_LLM_CALL)
    def log_after_llm(**ctx):
        r = ctx["response"]
        print(f"[hook:after_llm_call] stop_reason={r.stop_reason} "
              f"tool_calls={[tc.name for tc in r.tool_calls]}")

    @hooks.on(HookEvent.BEFORE_TOOL_CALL)
    def safety_guard(**ctx):
        # example of a hook that can block/rewrite a tool call before execution
        call: ToolCall = ctx["tool_call"]
        if call.name == "calculator":
            expr = call.arguments.get("expression", "")
            if "__" in expr or "import" in expr:
                blocked = ToolResult(call.id, call.name, "blocked: unsafe expression", is_error=True)
                print(f"[hook:before_tool_call] BLOCKED unsafe calculator call: {expr!r}")
                return {"abort": True, "result": blocked}
        print(f"[hook:before_tool_call] step={ctx['step']} name={call.name} args={call.arguments}")

    @hooks.on(HookEvent.AFTER_TOOL_CALL)
    def log_after_tool(**ctx):
        metrics["tool_calls"] += 1
        result: ToolResult = ctx["result"]
        print(f"[hook:after_tool_call] name={result.name} "
              f"is_error={result.is_error} content={result.content!r}")

    @hooks.on(HookEvent.ON_ERROR)
    def log_error(**ctx):
        print(f"[hook:on_error] {ctx.get('error')}")

    @hooks.on(HookEvent.ON_FINISH)
    def log_finish(**ctx):
        elapsed = time.time() - metrics["start_time"]
        print(f"[hook:on_finish] llm_calls={metrics['llm_calls']} "
              f"tool_calls={metrics['tool_calls']} elapsed={elapsed:.3f}s")

    hooks._metrics = metrics  # exposed for inspection in the demo below
    return hooks


# ============================================================
# 7. Demo entrypoint
# ============================================================

if __name__ == "__main__":
    use_real_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    llm: LLMClient = AnthropicLLMClient() if use_real_llm else ScriptedMockLLM()
    print(f"Using {'AnthropicLLMClient (real API)' if use_real_llm else 'ScriptedMockLLM (demo mode)'}\n")

    hooks = build_default_hooks()
    agent = Agent(
        llm=llm,
        tools=tools,
        hooks=hooks,
        system_prompt="You are a helpful assistant with access to a calculator, "
                       "a clock, and a fun-fact generator.",
    )

    query = "What's 23*19+4, what time is it, and tell me a fun fact about that number?"
    print("=" * 70)
    final = agent.run(query)
    print("=" * 70)
    print("FINAL ANSWER:", final)
