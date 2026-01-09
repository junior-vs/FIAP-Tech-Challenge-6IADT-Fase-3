import pytest

from src.domain.guardrails import GuardrailsValidator


@pytest.fixture()
def validator() -> GuardrailsValidator:
    return GuardrailsValidator()


def test_accepts_english_medical_question(validator: GuardrailsValidator):
    result = validator.validate_with_result("What is the treatment protocol for COPD?")
    assert result.is_valid is True


def test_rejects_english_non_medical_topic(validator: GuardrailsValidator):
    result = validator.validate_with_result("What is the best cake recipe?")
    assert result.is_valid is False
    assert result.is_medical_relevant is False


def test_does_not_flag_generic_patient_word_as_name(validator: GuardrailsValidator):
    result = validator.validate_with_result("Patient with fever and cough. What should be done?")
    assert result.is_valid is True


def test_flags_capitalized_patient_name(validator: GuardrailsValidator):
    result = validator.validate_with_result("Patient John Smith has fever. What should be done?")
    assert result.is_valid is False
    assert result.has_pii is True
