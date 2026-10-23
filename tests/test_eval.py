import json

from pocket_recall.eval import locomo
from pocket_recall.eval.run import _sign_test, main


def test_sign_test():
    assert _sign_test(0, 0) == 1.0
    assert abs(_sign_test(6, 0) - 0.03125) < 1e-12
    assert _sign_test(37, 2) < 1e-8 and _sign_test(37, 2) == _sign_test(2, 37)


def test_locomo_dates_use_longmemeval_format():
    assert locomo._date("1:56 pm on 8 May, 2023") == "2023/05/08 (Mon) 13:56"


def test_compare_pairs_only_same_relay(tmp_path, capsys):
    def write(name, rows):
        path = tmp_path / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    row = lambda q, relay, ok: {"question_id": q, "relay": relay, "category": "c", "correct": ok}
    a = write("a.jsonl", [row("1", "A", False), row("2", "A", True), row("3", "B", False)])
    b = write("b.jsonl", [row("1", "A", True), row("2", "A", True), row("3", "A", True)])
    main(["compare", "--a", str(a), "--b", str(b)])
    out = capsys.readouterr().out
    assert "Paired questions: 2" in out and "| c | 2 | 50.0% | 100.0% | 1 | 0 |" in out
