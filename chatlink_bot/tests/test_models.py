"""The same scenarios with REAL inference: Gemma reads each message and Laya
picks between candidates, through the Tachyon daemon. Only WhatsApp, SAGE and
the clock are faked. Slow (seconds per message) and needs a GPU:

    tachyon serve &          # with KAPA_CHAT_MODEL and KAPA_LAYA_MODEL pulled
    make test-models

These check what the fake tests cannot: that the models understand the way
clients write. A failure here is a recognition problem to look at, not
necessarily a bug. Add your clients' real messages as new cases.
"""
import pytest

from conftest import product

from chatlink_bot import models

pytestmark = pytest.mark.models

CATALOG = [
    product("KJ401", "KAJAL NEGRO INTENSO"),
    product("SN15", "SERUM NOCHE 15ML"),
    product("SN25", "SERUM NOCHE 25ML"),
    product("LB12", "LABIAL MATE ROJO PASION"),
]


def turn(k, text, at, reply_at):
    k.client_says(text, at=at)
    k.tick(reply_at)
    return k.last_reply() or ""


@pytest.fixture
def k(kapa_real):
    kapa_real.history["000139"] = CATALOG
    return kapa_real


def test_plain_order_with_quantity(k):
    r = turn(k, "Hola! ponme 6 del kajal negro porfa", "10:00", "10:16")
    assert ("KJ401", 6) in k.cart(), r
    assert "KJ401" in r


def test_informal_list_with_sizes(k):
    r = turn(k, "buenas, para esta semana: 6 kajal negro, 3 del serum de noche de 25 y 2 labiales rojos",
             "10:00", "10:16")
    assert set(k.cart()) >= {("KJ401", 6), ("SN25", 3), ("LB12", 2)}, (k.cart(), r)


def test_bulleted_list(k):
    r = turn(k, "Pedido:\n- kajal negro 6\n- serum noche 25ml 3", "10:00", "10:16")
    assert set(k.cart()) >= {("KJ401", 6), ("SN25", 3)}, (k.cart(), r)


def test_ambiguous_size_becomes_a_question(k):
    r = turn(k, "ponme 2 del serum de noche", "10:00", "10:16")
    assert not [c for c in k.cart() if c[0] in ("SN15", "SN25")], (k.cart(), r)
    assert "SN15" in r and "SN25" in r, r


def test_price_question_is_referred_to_the_salesman(k):
    r = turn(k, "a cuánto me dejas el kajal negro?", "10:00", "10:16")
    assert k.cart() == [], (k.cart(), r)
    assert "te informa" in r, r      # referral, not an invented price


def test_no_bot_opt_out(k):
    r = turn(k, "no quiero hablar con una máquina, que me escriba Ana", "10:00", "10:16")
    assert k.cart() == []
    assert not k.session().bot_enabled or "Ana" in r, r


def test_close_confirm_and_order_email(k):
    turn(k, "ponme 6 del kajal negro", "10:00", "10:16")
    r = turn(k, "eso es todo, mándalo", "10:20", "10:36")
    assert "¿Lo confirmo" in r, r
    r = turn(k, "sí, perfecto", "10:40", "10:56")
    assert "¡Pedido confirmado!" in r, r
    [o] = k.rows(models.KapaOrder)
    assert o.cart[0]["code"] == "KJ401" and o.email_status == "sent"


def test_no_withdraws_the_close_proposal(k):
    turn(k, "ponme 6 del kajal negro", "10:00", "10:16")
    turn(k, "ya está, ciérralo", "10:20", "10:36")
    r = turn(k, "no no, espera que miro una cosa", "10:40", "10:56")
    assert k.session().order_status != "CLOSED", r
    assert not [i for i in k.session().open_items if i["tipo"] == "propuesta_cierre"], r
