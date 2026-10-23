import pytest

from pocket_recall.redact import redact
from pocket_recall.store import MemoryStore

# Built at runtime so the repository itself never contains anything shaped like a real key.
FAKE = "Ab3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zA"


@pytest.mark.parametrize("secret, kind", [
    ("sk-ant-api03-" + FAKE, "anthropic-key"),
    ("sk-proj-" + FAKE, "openai-key"),
    ("ghp_" + FAKE + "1234", "github-token"),
    ("AKIA" + "IOSFODNN7EXAMPLE", "aws-access-key"),
    ("AIza" + FAKE + "abc", "google-api-key"),
    ("xoxb-" + "123456789012-" + FAKE, "slack-token"),
    ("hf_" + FAKE, "hf-token"),
    ("eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0." + FAKE, "jwt"),
])
def test_known_token_shapes(secret, kind):
    out = redact(f"my key is {secret} ok")
    assert secret not in out and f"[REDACTED:{kind}]" in out and out.endswith(" ok")


def test_assignments_urls_bearer_and_key_blocks():
    text = "\n".join([
        'ANTHROPIC_API_KEY=' + FAKE,
        '"client_secret": "' + FAKE + '"',
        "password: hunter2xyz9",
        "postgres://admin:s3cretPass@db.local:5432/app",
        "Authorization: Bearer " + FAKE,
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEow" + FAKE + "\n-----END RSA PRIVATE KEY-----",
    ])
    out = redact(text)
    assert FAKE not in out and "hunter2xyz9" not in out and "s3cretPass" not in out
    assert out.startswith("ANTHROPIC_API_KEY=[REDACTED:")
    assert "postgres://admin:[REDACTED:password]@db.local" in out


@pytest.mark.parametrize("text", [
    "max_tokens=4000 and token_count = 12345678",
    "tokenizer = AutoTokenizer.from_pretrained(name)",
    "api_key = os.environ['ANTHROPIC_API_KEY']",
    "password: changeme",
    "我的密码规则是至少 8 位",
    "sk-learn is a typo for scikit-learn",
])
def test_ordinary_text_is_kept(text):
    assert redact(text) == text


def test_store_masks_on_write_and_can_scrub_old_rows(tmp_path):
    store = MemoryStore(tmp_path / "m.db", scorer="bm25")
    store.add_message("user", "deploy with token sk-ant-api03-" + FAKE)
    assert FAKE not in store.recall("deploy token", 2000).text
    raw = MemoryStore(tmp_path / "old.db", scorer="bm25", redact_secrets=False)
    raw.add_message("user", "OPENAI_API_KEY=sk-" + FAKE)
    assert raw.redact_existing() == 1 and raw.redact_existing() == 0
    assert FAKE not in raw.export_markdown()
