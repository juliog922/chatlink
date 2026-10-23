# Kapa — specification

Kapa covers a busy salesman's WhatsApp and Gmail chats with his clients and
turns them into orders he only has to validate. This file is the spec: every
rule below is enforced by code, and the scenario IDs are test names in
`chatlink_bot/tests/test_scenarios.py`. It replaces `NOTES.md`.

Rule zero: no feature, setting or service exists unless a rule below needs it.

## Rules

1. **Coverage.** Kapa speaks only after the salesman has been silent for
   `RESPONSE_DELAY_MINUTES` (15) since the client's newest message, and never
   sends anything if the salesman wrote while the turn was running.
2. **Scope.** Kapa builds orders: add, change, remove, choose, repeat, recap,
   close. Price, stock, delivery, invoices, product details and complaints get
   a short referral (the salesman sees it in his own chat). Asking for the
   salesman gets "le aviso".
3. **Silence.** Small talk, a topic the salesman is already handling, and
   opted-out clients get no reply.
4. **Never invent.** A code in the cart comes from the client's own words
   resolved against his order history, his saved aliases, or a code he typed
   that exists in SAGE. A quantity is a number he wrote, or 1 when he wrote none.
5. **Do not overwhelm.** At most 2 product questions per reply
   (`PIPE_QUESTION_BUDGET`), at most 4 options per question; the rest wait.
6. **The salesman is the last filter.** An item still unresolved after 2 turns
   without progress, any item open at close, and everything after "no sé / lo
   que sea" becomes a review line: the client's own words, in the order email
   and the xlsx. Review lines never block a close.
7. **Explicit confirmation.** Closing, media orders, repeat orders and removals
   need a "sí"; a "no" withdraws the proposal.
8. **Nothing is lost.** Client messages are stored before anything else; a cart
   lives until it is closed; replies and closed orders are written with the
   session in one transaction and delivered with retries. Any failure that
   leaves a client or an order without an answer calls `alert()`.
9. **One identity.** A reply leaves from the salesman's own number or mailbox,
   or not at all.

## How a message flows

1. **Ingest** (`handlers.handle_new_message` / `handle_new_email`): admin
   commands, the salesman's own messages, and messages from SAGE clients are
   stored. Nothing else happens.
2. **Loop** (`logic/loop.py`, every 30 s): a conversation is due when it has
   client messages newer than both its cursor (`last_answered_id`) and the
   salesman's last message, and the newest is older than the delay. Due turns
   run one at a time; then queued replies and order emails are sent; then the
   heartbeat is pinged.
3. **Turn** (`handlers.run_turn` → `ai/llm.run_agent`): Gemma extracts typed
   acts; code resolves products (alias → order history → Laya when ambiguous);
   the reducer is the only writer of cart and open items; the reply is the
   template rendered from the reducer's facts.

## Scenarios: admin ↔ salesman

| ID | Who does what | System does |
| --- | --- | --- |
| A1 | Admin creates a salesman (`POST /api/users`, through an SSH tunnel) | User created, disabled |
| A2 | Salesman sends `login` to the admin number, or in self-chat | Pairing code by WhatsApp and email; Gmail listener starts |
| A4 | Salesman sends `logout` | Device unlinked, mailbox stopped, user disabled; carts kept |
| A5 | Registered user writes anything else to the admin number | Help card, at most every 10 s |
| A7 | meow reports logged out / stream replaced / temporary ban | User disabled, `alert()` to admin and salesman; client-outdated alerts the admin only |
| A8 | Gmail rejects the app password | `alert()` to admin and salesman; WhatsApp continues |
| A9 | Order closes | `pedido.xlsx` emailed to the salesman |
| A10 | A reply or an order email fails 3 times | Marked failed, `alert()` |

## Scenarios: client ↔ Kapa ↔ salesman

| ID | Who does what | When | Kapa does |
| --- | --- | --- | --- |
| C1, C2 | Client writes, maybe several times | Salesman enabled | Waits the delay from the newest message; answers all of them in one turn |
| C3 | Salesman writes (text, reaction, edit) | Any | Everything before it counts as answered |
| C4 | Salesman writes | During a turn | The turn is discarded: no reply, no cart change |
| C5 | Client writes | During a turn | Answered in the next turn |
| C6 | Service restarts | Messages pending | Nothing to restore: pending is computed from the table |
| C7 | Client writes in a group | Any | Ignored by meow |
| C8 | Client not in SAGE, or salesman disabled | Any | Not stored / stored, never answered |
| C9 | Phone matches several SAGE clients | Any | History off; `alert()` once so SAGE gets fixed |
| V1 | First message ever | Never introduced | Introduction plus the answer |
| V3 | Client comes back after 6 h | Cart has lines | Cart kept; "hola de nuevo" plus the recap |
| V4 | New order | Last order closed | Fresh cart |
| V5 | "No quiero hablar con un bot" | Any | Opt-out text, then silence; writing "Kapa" re-enables it |
| V6 | Small talk, or a topic the salesman handles | Any | Silence |
| V7 | Price, stock, details, complaint | Any | Short referral; order work in the same message still done |
| V8 | Asks for the salesman | Any | "Aviso a X para que se conecte" |
| P1 | Names a product he bought, one match | Any | Added |
| P2 | Several matches; Laya ≥ `LAYA_PICK` (0.6) | Any | Added |
| P3 | Several matches; Laya unsure | Any | Options question |
| P4 | Picks an option | Options pending | Added; his words saved as an alias (exact hit next time) |
| P5 | More than 2 items need a question | Any | 2 asked, "Me quedan N más por concretar" |
| P6 | "No sé", "lo que sea" | Products pending | They become review lines |
| K1 | Changes or removes a line | Line found | Applied; ≤ 0 asks first |
| K2 | Same product with another quantity | Any | Asks: new, old or sum |
| K3 | "Lo mismo que la última vez" | Kapa closed an order before | Proposal, added on "sí" |
| K4 | "Eso es todo" | Cart or pending items | Recap, pending items named, "¿Lo confirmo?" |
| K5 | "Sí" | Close proposal | CLOSED; order queued and emailed |
| K6 | "No" | Any yes/no proposal | Withdrawn: "Vale, lo dejo como estaba" |
| K7 | Closes with items unresolved | Any | They go to the salesman as review lines |
| M1 | Photo, audio or document of an order | Any | Read by Gemma (vision/speech) or the document parser; proposal, added on "sí" |
| F1 | Tachyon down | Any turn | No reply, `alert()`, retried every loop |

## Not built, until a measured failure asks for it

Granite embeddings, catalog search beyond the client's history, "assume and
tell" for medium confidence, learning from the orders the salesman enters in
SAGE, Laya for turn routing, a staging server, a permanent canary number, the
WhatsApp Cloud API. Each has its trigger in the v3 proposal.
