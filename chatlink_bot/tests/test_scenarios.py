"""One test per scenario row of the spec (SPEC.md). Times are UTC on one day;
RESPONSE_DELAY_MINUTES is 15."""
from conftest import CLIENT, SALESMAN, act, pedir, product

from chatlink_bot import models

KAJAL = [product("KJ401", "KAJAL NEGRO INTENSO")]
SERUMS = [product("SN15", "SERUM NOCHE 15ML"), product("SN25", "SERUM NOCHE 25ML")]


def order(kapa, at="10:00"):
    kapa.history["000139"] = KAJAL
    kapa.client_says("ponme 6 del kajal negro", at=at,
                     understood={"hablar": True, "actos": [pedir("kajal negro", 6, "6 del kajal negro")]})


# ------------------------------------------------------------------ coverage
def test_C1_C2_waits_for_the_delay_from_the_newest_message_and_batches(kapa):
    kapa.history["000139"] = KAJAL + SERUMS
    kapa.client_says("ponme 6 del kajal negro", at="10:00",
                     understood={"hablar": True, "actos": [pedir("kajal negro", 6, "6 del kajal negro")]})
    kapa.tick("10:10")
    kapa.client_says("y nada más de momento", at="10:12")
    kapa.tick("10:20")
    assert kapa.replies() == []            # 15 min counted from 10:12
    kapa.tick("10:28")
    assert len(kapa.replies()) == 1
    assert "Apuntado" in kapa.last_reply() and "KJ401" in kapa.last_reply()
    assert kapa.sent[-1].from_jid == f"{SALESMAN}:1@s.whatsapp.net"   # A7: his own device


def test_C3_salesman_reply_means_kapa_stays_silent(kapa):
    order(kapa)
    kapa.salesman_says("hola! ahora te lo preparo", at="10:05")
    kapa.tick("10:30")
    assert kapa.replies() == []


def test_C3_coverage_resumes_when_client_writes_again(kapa):
    order(kapa)
    kapa.salesman_says("te llamo luego", at="10:05")
    kapa.client_says("ok, y ponme 6 del kajal negro", at="10:10",
                     understood={"hablar": True, "actos": [pedir("kajal negro", 6, "6 del kajal negro")]})
    kapa.tick("10:26")
    assert len(kapa.replies()) == 1


def test_C4_reply_dropped_if_salesman_writes_during_the_turn(kapa, monkeypatch):
    order(kapa)
    original = kapa.run.__self__  # noqa: F841 (keeps the harness alive)
    from chatlink_bot import handlers
    real = handlers.run_agent

    async def slow_agent(**kw):
        result = await real(**kw)
        await handlers.handle_new_message({"normalized": __import__("types").SimpleNamespace(
            from_phone=SALESMAN, to_phone=CLIENT, text="ya estoy aquí", filename="", binary=b"",
            from_jid=f"{SALESMAN}:1@s.whatsapp.net", to_jid=f"{CLIENT}@s.whatsapp.net")})
        return result
    monkeypatch.setattr(handlers, "run_agent", slow_agent)
    kapa.tick("10:16")
    assert kapa.replies() == []
    assert kapa.cart() == []               # his turn, his cart: nothing applied


def test_C5_message_during_a_turn_is_answered_next(kapa, monkeypatch):
    order(kapa)
    from chatlink_bot import handlers
    real = handlers.run_agent
    calls = []

    async def agent(**kw):
        calls.append(kw["current_message"])
        result = await real(**kw)
        if len(calls) == 1:   # the client writes while the first turn runs
            kapa.extractions.append({"hablar": True, "actos": [pedir("kajal negro", 2, "2 kajal negro")]})
            await handlers.handle_new_message({"normalized": __import__("types").SimpleNamespace(
                from_phone=CLIENT, to_phone=SALESMAN, text="y 2 kajal negro más", filename="", binary=b"",
                from_jid=f"{CLIENT}@s.whatsapp.net", to_jid=f"{SALESMAN}:1@s.whatsapp.net")})
        return result
    monkeypatch.setattr(handlers, "run_agent", agent)
    kapa.tick("10:16")
    kapa.tick("10:40")
    assert calls == ["ponme 6 del kajal negro", "y 2 kajal negro más"]
    assert len(kapa.replies()) == 2


def test_C6_nothing_to_restore_after_restart(kapa):
    order(kapa, at="10:00")
    # no tick before the "restart": a later tick still finds it
    kapa.tick("11:00")
    assert len(kapa.replies()) == 1


def test_C8_client_not_in_sage_is_never_stored(kapa):
    kapa.client_says("hola", at="10:00", phone="34699999999")
    kapa.tick("10:30")
    assert kapa.rows(models.Chat) == []


def test_C9_ambiguous_phone_alerts_once(kapa):
    kapa.clients[CLIENT] = ("000139", "Perfumería Luz", False)
    kapa.client_says("hola", at="10:00", understood={"hablar": True, "actos": [act("otro", "hola", subtipo="saludo")]})
    kapa.tick("10:16")
    assert any(k.startswith("ambiguous:") for k, _ in kapa.alerts)


# ------------------------------------------------------------- conversation
def test_V1_first_message_gets_the_introduction(kapa):
    order(kapa)
    kapa.tick("10:16")
    assert kapa.last_reply().startswith("¡Hola, Perfumería! Soy Kapa")


def test_V3_cart_survives_six_hours_of_silence(kapa):
    order(kapa)
    kapa.tick("10:16")
    kapa.client_says("hola", at="17:30", understood={"hablar": True, "actos": [act("otro", "hola", subtipo="saludo")]})
    kapa.tick("17:46")
    assert kapa.cart() == [("KJ401", 6)]
    assert "llevas apuntado" in kapa.last_reply()


def test_V5_opt_out_then_silence_then_kapa_by_name(kapa):
    kapa.client_says("no quiero hablar con un bot", at="10:00",
                     understood={"hablar": True, "actos": [act("otro", "no quiero hablar con un bot", subtipo="rechazo_bot")]})
    kapa.tick("10:16")
    assert "no te escribiré más" in kapa.last_reply()
    kapa.client_says("ponme 6 del kajal negro", at="11:00")
    kapa.tick("11:16")
    assert len(kapa.replies()) == 1
    order(kapa, at="12:00")
    kapa.client_says("Kapa, ponme el kajal", at="12:01",
                     understood={"hablar": True, "actos": [pedir("kajal negro", 6, "6 del kajal negro")]})
    kapa.tick("12:20")
    assert len(kapa.replies()) == 2


# ------------------------------------------------------------- recognition
def test_P2_laya_confident_pick_is_added(kapa):
    kapa.history["000139"] = SERUMS
    kapa.laya.append({"q0": ("1", 0.92)})
    kapa.client_says("ponme 3 serum noche", at="10:00",
                     understood={"hablar": True, "actos": [pedir("serum noche", 3, "3 serum noche")]})
    kapa.tick("10:16")
    assert kapa.cart() == [("SN25", 3)]


def test_P3_laya_unsure_becomes_a_question(kapa):
    kapa.history["000139"] = SERUMS
    kapa.client_says("ponme 3 serum noche", at="10:00",
                     understood={"hablar": True, "actos": [pedir("serum noche", 3, "3 serum noche")]})
    kapa.tick("10:16")
    assert kapa.cart() == []
    assert "SN15" in kapa.last_reply() and "SN25" in kapa.last_reply() and "¿Cuál quieres?" in kapa.last_reply()


def test_P5_question_budget_asks_two_and_counts_the_rest(kapa):
    kapa.history["000139"] = []
    msg = "ponme pinza roja, lima azul, borla grande y esponja rosa"
    kapa.client_says(msg, at="10:00", understood={"hablar": True, "actos": [
        pedir("pinza roja"), pedir("lima azul"), pedir("borla grande"), pedir("esponja rosa")]})
    kapa.tick("10:16")
    r = kapa.last_reply()
    assert r.count("no lo encuentro") == 2
    assert "Me quedan 2 más por concretar" in r


def test_P4_alias_learned_from_a_pick_is_an_exact_hit_next_time(kapa):
    kapa.history["000139"] = SERUMS
    kapa.client_says("ponme 3 serum noche", at="10:00",
                     understood={"hablar": True, "actos": [pedir("serum noche", 3, "3 serum noche")]})
    kapa.tick("10:16")
    kapa.client_says("SN25", at="10:20",
                     understood={"hablar": True, "actos": [act("elegir", "SN25", objetivo_texto="SN25")]})
    kapa.tick("10:36")
    assert kapa.cart() == [("SN25", 3)]
    assert [(a.alias, a.code) for a in kapa.rows(models.ClientAlias)] == [("serum noche", "SN25")]
    calls = kapa.laya_calls
    kapa.client_says("y otros 2 serum noche", at="11:00",
                     understood={"hablar": True, "actos": [pedir("serum noche", 2, "2 serum noche")]})
    kapa.tick("11:16")
    assert kapa.laya_calls == calls        # no choice needed any more
    # Same product with a new quantity: the line stays and Kapa asks (K2).
    assert kapa.cart() == [("SN25", 3)]
    assert "¿Con cuál me quedo" in kapa.last_reply() and "2 , 3 , 5" in kapa.last_reply()


# -------------------------------------------------------- changes and close
def close_order(kapa):
    order(kapa)
    kapa.tick("10:16")
    kapa.client_says("eso es todo", at="10:20",
                     understood={"hablar": True, "actos": [act("cerrar", "eso es todo")]})
    kapa.tick("10:36")


def test_K4_K5_close_queues_the_order_and_emails_it(kapa):
    close_order(kapa)
    assert "¿Lo confirmo" in kapa.last_reply()
    kapa.client_says("sí", at="10:40", understood={"hablar": True, "actos": [act("confirmar", "sí")]})
    kapa.tick("10:56")
    assert "¡Pedido confirmado!" in kapa.last_reply()
    [o] = kapa.rows(models.KapaOrder)
    assert o.email_status == "sent" and o.cart[0]["code"] == "KJ401"
    [mail] = kapa.emails
    assert mail.to == "ana@kapa.test" and mail.attachments[0][0] == "pedido.xlsx"


def test_K6_no_withdraws_the_proposal(kapa):
    close_order(kapa)
    kapa.client_says("no, espera", at="10:40", understood={"hablar": True, "actos": [act("negar", "no, espera")]})
    kapa.tick("10:56")
    assert "lo dejo como estaba" in kapa.last_reply()
    assert kapa.session().order_status == "BUILDING"
    assert not [i for i in kapa.session().open_items if i["tipo"] == "propuesta_cierre"]


def test_K7_unresolved_products_go_to_review_and_never_block_the_close(kapa):
    kapa.history["000139"] = KAJAL
    kapa.client_says("ponme 6 del kajal negro y 2 pinzas doradas", at="10:00", understood={"hablar": True, "actos": [
        pedir("kajal negro", 6, "6 del kajal negro"), pedir("pinzas doradas", 2, "2 pinzas doradas")]})
    kapa.tick("10:16")
    kapa.client_says("eso es todo", at="10:20", understood={"hablar": True, "actos": [act("cerrar", "eso es todo")]})
    kapa.tick("10:36")
    kapa.client_says("sí", at="10:40", understood={"hablar": True, "actos": [act("confirmar", "sí")]})
    kapa.tick("10:56")
    [o] = kapa.rows(models.KapaOrder)
    assert o.review == [{"texto": "pinzas doradas", "qty": 2}]
    assert "pinzas doradas" in kapa.last_reply()


def test_P6_no_sabe_turns_pending_questions_into_review_lines(kapa):
    kapa.history["000139"] = []
    kapa.client_says("ponme pinzas doradas", at="10:00", understood={"hablar": True, "actos": [pedir("pinzas doradas")]})
    kapa.tick("10:16")
    kapa.client_says("no sé, lo que diga Ana", at="10:20",
                     understood={"hablar": True, "actos": [act("otro", "no sé", subtipo="no_sabe")]})
    kapa.tick("10:36")
    assert "para que lo revise: pinzas doradas" in kapa.last_reply()
    assert [i["tipo"] for i in kapa.session().open_items] == ["revisar"]


# ----------------------------------------------------------------- failures
def test_F1_inference_down_means_silence_alert_and_retry(kapa):
    order(kapa)
    kapa.inference_up = False
    kapa.tick("10:16")
    assert kapa.replies() == [] and any(k.startswith("turn:") for k, _ in kapa.alerts)
    kapa.inference_up = True
    kapa.extractions.append({"hablar": True, "actos": [pedir("kajal negro", 6, "6 del kajal negro")]})
    kapa.tick("10:17")
    assert len(kapa.replies()) == 1


def test_A10_send_fails_three_times_then_alerts(kapa):
    order(kapa)
    kapa.send_ok = False
    for t in ("10:16", "10:17", "10:18"):
        kapa.tick(t)
    [row] = [r for r in kapa.rows(models.Chat) if r.is_bot]
    assert row.status == "failed" and row.attempts == 3
    assert any(k.startswith("send:") for k, _ in kapa.alerts)


def test_A10_order_email_retries_then_alerts(kapa):
    close_order(kapa)
    kapa.email_ok = False
    kapa.client_says("sí", at="10:40", understood={"hablar": True, "actos": [act("confirmar", "sí")]})
    for t in ("10:56", "10:57", "10:58"):
        kapa.tick(t)
    [o] = kapa.rows(models.KapaOrder)
    assert o.email_status == "failed"
    assert any(k.startswith("order:") for k, _ in kapa.alerts)


def test_A7_device_logged_out_disables_the_salesman_and_alerts(kapa):
    kapa.run(__import__("chatlink_bot").handlers.handle_device_event(SALESMAN, "logged_out"))
    [u] = kapa.rows(models.User)
    assert not u.enabled and u.wa_device_jid is None
    assert kapa.alerts and kapa.alerts[0][0] == f"device:{SALESMAN}:logged_out"
    order(kapa)
    kapa.tick("10:16")
    assert kapa.replies() == []            # C8: stored, never answered


def test_A2_admin_login_command_never_reaches_the_ai(kapa):
    seen = []
    from chatlink_bot.events import event_bus
    kapa.run(event_bus.subscribe("admin_command", lambda p: seen.append(p)))
    # registered user writes `login` in self-chat
    kapa._wa(SALESMAN, SALESMAN, "login")
    assert seen and seen[0]["command"] == "login"
    assert kapa.rows(models.Chat) == []
