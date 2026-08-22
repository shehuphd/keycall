import shutil
import subprocess

import pytest

from keycall._sources import SourceError, load_targets

CANARY = "sk-canary-source-key-000"

no_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_txt_multiple_targets_with_comments_and_quotes(tmp_path):
    source = write(
        tmp_path,
        "keys.txt",
        f"""
# testing keys
protocol=openai provider=openai key={CANARY} name=OPENAI_TESTING_KEY
protocol=anthropic provider=anthropic key=sk-ant-x name=claude-test
protocol=openai-compatible provider=university-lab base_url=https://llm.example.edu/v1 key=AQ-x name="mY_UniversitY-TesT1ng_Key"
""",
    )
    targets, _ = load_targets(source)
    assert len(targets) == 3
    assert targets[0].provider == "openai"
    assert targets[0].key == CANARY
    assert targets[2].name == "mY_UniversitY-TesT1ng_Key"  # quotes discarded, casing kept
    assert targets[2].base_url == "https://llm.example.edu/v1"


def test_txt_escaped_quotes_stay_literal(tmp_path):
    source = write(
        tmp_path, "keys.txt", 'provider=openai key=k name="say \\"hi\\" now"\n'
    )
    targets, _ = load_targets(source)
    assert targets[0].name == 'say "hi" now'


def test_txt_duplicate_field_rejected(tmp_path):
    source = write(tmp_path, "keys.txt", "provider=openai provider=openai key=k\n")
    with pytest.raises(SourceError, match="duplicate"):
        load_targets(source)


def test_txt_malformed_record_rejected_without_leaking_values(tmp_path):
    source = write(tmp_path, "keys.txt", f"provider=openai key={CANARY} $(rm -rf /)\n")
    with pytest.raises(SourceError) as excinfo:
        load_targets(source)
    assert CANARY not in str(excinfo.value)
    assert "line 1" in str(excinfo.value)


def test_txt_missing_required_field_error_names_field_not_value(tmp_path):
    source = write(tmp_path, "keys.txt", "provider=openai name=x\n")
    with pytest.raises(SourceError, match="key"):
        load_targets(source)


def test_txt_command_like_payload_stays_inert_data(tmp_path):
    source = write(tmp_path, "keys.txt", 'provider=openai key="$(whoami)"\n')
    targets, _ = load_targets(source)
    # Parsed as an opaque string, never evaluated.
    assert targets[0].key == "$(whoami)"


def test_json_targets(tmp_path):
    source = write(
        tmp_path,
        "keys.json",
        f'{{"targets": [{{"provider": "openai", "key": "{CANARY}", "name": "t1"}}]}}',
    )
    targets, _ = load_targets(source)
    assert targets[0].provider == "openai"
    assert targets[0].key == CANARY


def test_toml_targets(tmp_path):
    source = write(
        tmp_path,
        "keys.toml",
        f"""
[[targets]]
provider = "anthropic"
key = "{CANARY}"
name = "claude-test"
""",
    )
    targets, _ = load_targets(source)
    assert targets[0].provider == "anthropic"
    assert targets[0].name == "claude-test"


def test_unknown_field_rejected(tmp_path):
    source = write(tmp_path, "keys.txt", "provider=openai key=k model=gpt-4o\n")
    with pytest.raises(SourceError, match="unknown field"):
        load_targets(source)


def test_env_source(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", CANARY)
    targets, warnings = load_targets("env:MY_TEST_KEY", provider="openai")
    assert targets[0].key == CANARY
    assert targets[0].provider == "openai"
    assert targets[0].name == "MY_TEST_KEY"
    assert warnings == []


def test_env_source_requires_provider(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", CANARY)
    with pytest.raises(SourceError, match="--provider"):
        load_targets("env:MY_TEST_KEY")


def test_env_source_missing_variable(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(SourceError, match="NOPE_KEY"):
        load_targets("env:NOPE_KEY", provider="openai")


def test_broadly_readable_file_warns(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text(f"provider=openai key={CANARY}\n", encoding="utf-8")
    path.chmod(0o644)
    _, warnings = load_targets(str(path))
    assert any("readable" in w.message for w in warnings)
    path.chmod(0o600)
    _, warnings = load_targets(str(path))
    assert not any("readable" in w.message for w in warnings)


def test_no_git_repo_stays_silent(tmp_path):
    # No .git anywhere above this file: the old check keyed off directory
    # presence alone and could never distinguish this from the git cases
    # below, so this stays its own explicit case.
    path = write(tmp_path, "keys.txt", f"provider=openai key={CANARY}\n")
    _, warnings = load_targets(path)
    assert not any("git" in w.message for w in warnings)


@no_git
def test_git_tracked_file_warns_strongly(tmp_path):
    git("init", cwd=tmp_path)
    path = write(tmp_path, "keys.txt", f"provider=openai key={CANARY}\n")
    git("add", "keys.txt", cwd=tmp_path)
    _, warnings = load_targets(path)
    assert any("tracked by git" in w.message for w in warnings)


@no_git
def test_git_untracked_and_unignored_file_warns(tmp_path):
    git("init", cwd=tmp_path)
    path = write(tmp_path, "keys.txt", f"provider=openai key={CANARY}\n")
    _, warnings = load_targets(path)
    assert any("git working tree" in w.message for w in warnings)
    assert not any("tracked by git" in w.message for w in warnings)


@no_git
def test_git_ignored_file_stays_silent(tmp_path):
    git("init", cwd=tmp_path)
    (tmp_path / ".gitignore").write_text("keys.txt\n", encoding="utf-8")
    path = write(tmp_path, "keys.txt", f"provider=openai key={CANARY}\n")
    _, warnings = load_targets(path)
    assert not any("git" in w.message for w in warnings)


def test_empty_source_rejected(tmp_path):
    source = write(tmp_path, "keys.txt", "# only a comment\n")
    with pytest.raises(SourceError, match="no targets"):
        load_targets(source)


def test_control_characters_removed_from_display_name(tmp_path):
    source = write(
        tmp_path, "keys.txt", 'provider=openai key=k name="evil\\x1b]0;pwned\x07name"\n'
    )
    targets, _ = load_targets(source)
    assert "\x07" not in targets[0].display_name
    assert "\x1b" not in targets[0].display_name
