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


# --- named-field container (key/secret pairs) --------------------------------

SECRET = "livekit-secret-abc123def456"


def test_mapping_reveals_named_fields():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    assert cred.reveal() == CANARY  # primary is api_key
    assert cred.reveal("api_key") == CANARY
    assert cred.reveal("api_secret") == SECRET


def test_mapping_must_carry_api_key():
    with pytest.raises(ValueError):
        Credential({"api_secret": SECRET})


def test_mapping_blank_field_rejected():
    for bad in ({"api_key": CANARY, "api_secret": ""}, {"api_key": "   ", "api_secret": SECRET}):
        with pytest.raises(ValueError):
            Credential(bad)


def test_non_string_non_mapping_rejected():
    for bad in (None, 42, ["k", "v"]):
        with pytest.raises(TypeError):
            Credential(bad)  # type: ignore[arg-type]


def test_reveal_unknown_field_names_it_without_leaking():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    with pytest.raises(ValueError) as excinfo:
        cred.reveal("api_token")
    assert "api_token" in str(excinfo.value)
    assert CANARY not in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


def test_secret_values_covers_every_field():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    assert set(cred.secret_values()) == {CANARY, SECRET}
    assert Credential(CANARY).secret_values() == (CANARY,)


def test_has_field():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    assert cred.has_field("api_secret")
    assert not cred.has_field("api_token")
    assert not Credential(CANARY).has_field("api_secret")


def test_pair_repr_str_format_hide_every_field():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    for rendered in (repr(cred), str(cred), f"{cred}", format(cred, ">40")):
        assert CANARY not in rendered
        assert SECRET not in rendered
        assert "redacted" in rendered


def test_pair_pickle_and_copy_blocked():
    cred = Credential({"api_key": CANARY, "api_secret": SECRET})
    with pytest.raises(TypeError):
        pickle.dumps(cred)
    with pytest.raises(TypeError):
        copy.copy(cred)
    with pytest.raises(TypeError):
        copy.deepcopy(cred)


def test_pair_fingerprint_keyed_on_primary_only():
    # api_secret does not enter the cache identity; two clients that share a
    # key differ only where the model-list cache never reaches.
    a = Credential({"api_key": CANARY, "api_secret": SECRET})
    b = Credential({"api_key": CANARY, "api_secret": "different-secret-000"})
    assert a.fingerprint() == b.fingerprint()
    assert CANARY not in a.fingerprint()
    assert SECRET not in a.fingerprint()
