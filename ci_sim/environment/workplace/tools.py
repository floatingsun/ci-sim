"""Typed definitions for the simulated workplace write tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from ci_sim.contracts import StrictModel


class GmailSendArgs(StrictModel):
    to: str
    subject: str
    body: str
    cc: str | list[str] | None = None
    bcc: str | list[str] | None = None


class SlackPostArgs(StrictModel):
    channel: str
    text: str


class CalendarCreateEventArgs(StrictModel):
    title: str
    start: str
    end: str
    attendees: list[str] = Field(min_length=1)
    description: str
    location: str | None = None
    timezone: str | None = None
    recurrence: str | list[str] | None = None
    calendar_id: str | None = None
    visibility: str | None = None


class DocsCreateArgs(StrictModel):
    title: str
    body: str


class DriveShareArgs(StrictModel):
    file_id: str
    recipients: list[str] = Field(min_length=1)
    role: Literal["viewer", "commenter", "editor"]
    message: str | None = None


@dataclass(frozen=True)
class WorkplaceTool:
    canonical_name: str
    model_name: str
    args_model: type[StrictModel]
    success_status: str
    id_prefix: str

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.args_model.model_validate(arguments).model_dump(exclude_none=True)


WORKPLACE_TOOLS = (
    WorkplaceTool(
        canonical_name="gmail.send",
        model_name="gmail_send",
        args_model=GmailSendArgs,
        success_status="sent",
        id_prefix="email",
    ),
    WorkplaceTool(
        canonical_name="slack.post",
        model_name="slack_post",
        args_model=SlackPostArgs,
        success_status="posted",
        id_prefix="slack",
    ),
    WorkplaceTool(
        canonical_name="calendar.create_event",
        model_name="calendar_create_event",
        args_model=CalendarCreateEventArgs,
        success_status="created",
        id_prefix="event",
    ),
    WorkplaceTool(
        canonical_name="docs.create",
        model_name="docs_create",
        args_model=DocsCreateArgs,
        success_status="created",
        id_prefix="doc",
    ),
    WorkplaceTool(
        canonical_name="drive.share",
        model_name="drive_share",
        args_model=DriveShareArgs,
        success_status="shared",
        id_prefix="share",
    ),
)

TOOLS_BY_MODEL_NAME = {tool.model_name: tool for tool in WORKPLACE_TOOLS}
TOOLS_BY_CANONICAL_NAME = {tool.canonical_name: tool for tool in WORKPLACE_TOOLS}
