# chatlink_bot/src/chatlink_bot/logic/lifecycle.py
"""
Conversation lifecycle for the Order Assistant — pure functions, stdlib only.

The main-goal invariant: the LLM decides the CLIENT'S INTENT, code decides the
CONVERSATION LIFECYCLE. A conversation ENDS when an order is dispatched
(CLOSED), when the client opts out, or when the client goes silent past the
session gap. The next COMMERCIAL intent after an ended conversation starts a
NEW conversation — fresh cart, Kapa introduces itself again — even minutes
after a close. Pleasantries ("gracias 😊") after an ended conversation get a
cordial reply but do NOT reopen it, so the following order intent still gets
the introduction.

Modes handed to the agent prompt:
  new      first contact ever            -> unconditional introduction
  renew    previous conversation ended   -> introduce only on commercial intent
  ongoing  conversation open             -> introduction forbidden
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def conversation_mode(*, introduced: bool, conv_open: bool, order_status: str,
                      gap_expired: bool) -> tuple[str, bool]:
    """-> (intro_mode, ended). `ended` also means: work on a fresh cart."""
    ended = (not conv_open) or order_status == "CLOSED" or gap_expired
    mode = "new" if not introduced else ("renew" if ended else "ongoing")
    return mode, ended


@dataclass(frozen=True)
class TurnOutcome:
    """Session updates after one agent turn. conv_open=None means keep as-is (silence)."""
    commercial: bool
    conv_open: Optional[bool]
    introduced_now: bool


def lifecycle_after_turn(*, intro_mode: str, spoke_as_kapa: bool, result_status: str,
                         cart_size: int, handoff: bool, opt_out: bool) -> TurnOutcome:
    """
    Rules:
    - Order work (items in cart, or status BUILDING/CLOSED) = commercial engagement.
    - CLOSED or opt-out ends the conversation.
    - "renew" without commercial engagement stays ended -> the next order
      intent still triggers a re-introduction (spec: a new commercial intent
      after a goodbye/close is a new conversation, gap or no gap).
    - Otherwise a delivered Kapa reply (or order work) opens/keeps it open.
    - "Introduced" is recorded only when Kapa actually greeted: any delivered
      reply in "new" mode; a commercial re-start in "renew" mode. Canned
      handoff/opt-out texts are never introductions (spoke_as_kapa=False).
    """
    commercial = cart_size > 0 or result_status in ("BUILDING", "CLOSED")
    if result_status == "CLOSED" or opt_out:
        conv_open: Optional[bool] = False
    elif intro_mode == "renew" and not commercial:
        conv_open = False
    elif spoke_as_kapa or commercial:
        conv_open = True
    else:
        conv_open = None  # silence: keep previous state
    introduced_now = spoke_as_kapa and (
        intro_mode == "new" or (intro_mode == "renew" and commercial))
    return TurnOutcome(commercial=commercial, conv_open=conv_open, introduced_now=introduced_now)