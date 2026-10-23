"""Pure functions: no database, no fakes."""
import io
import wave

from chatlink_bot.ai import inference
from chatlink_bot.ai.parsers import _wav_slices


def test_json_from_tolerates_fences_and_chatter():
    assert inference.json_from('Claro:\n```json\n{"hablar": true, "actos": []}\n```') == {"hablar": True, "actos": []}
    assert inference.json_from("nada") is None


def test_choose_applies_the_laya_threshold(monkeypatch):
    def decide(state, questions, timeout=30):
        return {"q0": {"choice": "1", "probabilities": {"0": 0.05, "1": 0.9, "ninguno": 0.05}},
                "q1": {"choice": "0", "probabilities": {"0": 0.45, "1": 0.4, "ninguno": 0.15}},
                "q2": {"choice": "ninguno", "probabilities": {"0": 0.1, "ninguno": 0.9}}}
    monkeypatch.setattr(inference, "decide", decide)
    out = inference.choose("msg", [{"texto": "a", "opciones": ["A", "B"]},
                                   {"texto": "b", "opciones": ["C", "D"]},
                                   {"texto": "c", "opciones": ["E"]}])
    # Laya's "ninguno" over the client's own candidates becomes a question, never "not found".
    assert [(o["veredicto"], o["indice"]) for o in out] == [("unico", 1), ("varios", None), ("varios", None)]


def test_wav_slices_cut_long_audio_into_30s_pieces():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * 70)
    slices = _wav_slices(buf.getvalue(), 30)
    assert len(slices) == 3
    assert all(wave.open(io.BytesIO(s)).getnframes() <= 16000 * 30 for s in slices)


def test_bullet_line_quantities():
    from chatlink_bot.ai.llm import _parse_linea_pedido
    assert _parse_linea_pedido("- kajal negro 6") == ("kajal negro", 6)
    assert _parse_linea_pedido("- serum noche 25ml 3") == ("serum noche 25ml", 3)
    assert _parse_linea_pedido("- kajal negro x12") == ("kajal negro", 12)
    assert _parse_linea_pedido("- KAJAL 401") == ("KAJAL 401", None)   # 3 digits: a reference


def test_plural_in_es_meets_singular():
    from conftest import product
    from chatlink_bot.ai.rag import client_history
    r = client_history.match("labiales rojos", [product("LB12", "LABIAL MATE ROJO PASION")], limit=5)
    assert r.verdict == "resolved"


def test_price_question_is_not_an_order():
    from chatlink_bot.ai.llm import Acto, Cantidad, validar_actos
    [a] = validar_actos([Acto(tipo="pedir", producto_texto="kajal negro", evidencia="el kajal negro")],
                        "a cuánto me dejas el kajal negro?")
    assert (a.tipo, a.subtipo) == ("otro", "precio_stock_factura")
    [a] = validar_actos([Acto(tipo="pedir", producto_texto="kajal negro", evidencia="6 del kajal negro",
                              cantidad=Cantidad(modo="total", valor=6))], "ponme 6 del kajal negro")
    assert (a.tipo, a.cantidad.valor) == ("pedir", 6)
