from __future__ import annotations

from textual.message import Message
from textual.widgets import Button, Static

from vibe.core.planner import PlanDecision, PlanState, PlanStep, PlanStepStatus


class PlannerPanel(Static):
    class DecisionSelected(Message):
        def __init__(self, decision_id: str, selection: str) -> None:
            super().__init__()
            self.decision_id = decision_id
            self.selection = selection

    def __init__(self) -> None:
        super().__init__("", id="planner-panel")

    async def update_plan(self, plan: PlanState | None) -> None:
        await self.remove_children()
        if not plan:
            await self.mount(
                Static("Planning inactive. Run `/plan <goal>` to start.", classes="planner-placeholder")
            )
            return

        status = plan.status.value.replace("_", " ").title()
        await self.mount(
            Static(f"Plan: {plan.goal}", classes="planner-title")
        )
        await self.mount(
            Static(f"Status: {status}", classes="planner-status")
        )
        if not plan.steps:
            await self.mount(
                Static("No steps yet.", classes="planner-placeholder")
            )
        else:
            for step in plan.steps:
                await self.mount(self._render_step(step))

        if plan.decisions:
            await self.mount(
                Static("Decisions", classes="planner-decisions-header")
            )
            for decision in plan.decisions:
                await self.mount(self._render_decision(decision))

    def _render_step(self, step: PlanStep) -> Static:
        status_class = f"planner-step-status-{step.status.value.lower()}"
        owner = f" · owner: {step.owner}" if step.owner else ""
        mode = f" · mode: {step.mode}" if step.mode else ""
        notes = f"\n{step.notes}" if step.notes else ""
        content = f"[{self._format_status(step.status)}]{owner}{mode} {step.title}{notes}"
        return Static(content, classes=f"planner-step {status_class}")

    def _render_decision(self, decision: PlanDecision) -> Static:
        status = "resolved" if decision.resolved else "pending"
        container = Static(classes=f"planner-decision planner-decision-{status}")
        content = f"{decision.decision_id}: {decision.question}"
        if decision.options:
            content += "\nOptions: " + ", ".join(decision.options)
        if decision.selection:
            content += f"\nChosen: {decision.selection}"
        container.update(content)
        if not decision.resolved:
            if decision.options:
                for option in decision.options:
                    button = Button(option, classes="planner-decision-button", id=f"{decision.decision_id}:{option}")
                    container.mount(button)
            else:
                button = Button("Enter decision in chat", classes="planner-decision-button")
                container.mount(button)
        return container

    def _format_status(self, status: PlanStepStatus) -> str:
        return status.value.replace("_", " ").title()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if ":" not in button_id:
            return
        decision_id, selection = button_id.split(":", 1)
        self.post_message(self.DecisionSelected(decision_id, selection))
