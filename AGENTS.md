# colibri-laguna — subagent & orchestration protocol

The opencode free-tier gateway (`opencode/deepseek-v4-flash-free`) intermittently
drops a subagent's FINAL response stream (e.g. `[503] The request queue is full`).
When that happens the parent's Task tool call returns an EMPTY `<task_result>` —
no error, no retry (opencode issues #38866 and #40527, both still open upstream).
Work the subagent already did is kept; only its final report is lost. This file
makes subagent runs survive that failure.

## Orchestrators (the agent dispatching Task tool calls)

Every Task you dispatch MUST paste the following block into the Task prompt (this
matters because subagents running inside `.worktrees/phaseN` do not see this file):

> SUBAGENT PROTOCOL:
> 1. Take ONE concrete action per turn (read/edit/write/bash). Never reply with a
>    plan-only message; execute the first step immediately.
> 2. Before your final turn, persist a short markdown report of what you did into a
>    file in the repo (e.g. `docs/.reports/<slug>.md`) and name it in your summary.
> 3. End with a SHORT plain-text summary (<200 words): what you changed, what file(s)
>    you wrote, the next step. Never end on a reasoning/thinking-only turn or a plan.
> 4. If this prompt is a resume of a prior session, continue from where it stopped —
>    do not re-plan from scratch.

## Subagents (any run via the Task tool)

1. ACT, don't plan. Execute the first step of the task in the same turn you receive
   the prompt. A plan-only reply wastes a stream turn and risks a lost result.
2. Persist durable results to disk BEFORE your final turn (report files, not just
   chat text) so the orchestrator can recover them even if your stream dies.
3. Your final turn must be a short plain-text summary (1–3 lines). No reasoning-only
   final turn, no plan, no giant markdown block.
4. If you are resumed (same task re-sent), continue from where you stopped.
