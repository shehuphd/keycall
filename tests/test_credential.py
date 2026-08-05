import copy
import pickle

import pytest

from keycall._credential import Credential

CANARY = "sk-canary-9f8e7d6c5b4a3210"


def test_reveal_returns_value():
    assert Credential(CANARY).reveal() == CANARY


def test_repr_str_format_never_contain_value():
    cred = Credential(CANARY)
    for rendered in (repr(cred), str(cred), f"{cred}", f"{cred!r}", f"{cred!s}", format(cred, ">40")):
        assert CANARY not in rendered
        assert "redacted" in rendered


def test_exception_context_never_contains_value():
    cred = Credential(CANARY)
    try:
        raise RuntimeError(f"failed handling {cred}")
    except RuntimeError as exc:
        assert CANARY not in str(exc)


def test_pickle_blocked():
    with pytest.raises(TypeError):
        pickle.dumps(Credential(CANARY))


def test_copy_and_deepcopy_blocked():
    cred = Credential(CANARY)
    with pytest.raises(TypeError):
        copy.copy(cred)
    with pytest.raises(TypeError):
        copy.deepcopy(cred)


def test_empty_or_blank_rejected():
    with pytest.raises(ValueError):
        Credential("")
    with pytest.raises(ValueError):
        Credential("   ")


def test_fingerprint_stable_within_process_and_not_the_value():
    a = Credential(CANARY)
    b = Credential(CANARY)
    assert a.fingerprint() == b.fingerprint()
    assert CANARY not in a.fingerprint()
    assert a.fingerprint() != Credential("sk-other-key-000").fingerprint()


def test_no_dict_and_no_public_value_attribute():
    cred = Credential(CANARY)
    assert not hasattr(cred, "__dict__")
    assert not hasattr(cred, "value")
    assert not hasattr(cred, "api_key")
