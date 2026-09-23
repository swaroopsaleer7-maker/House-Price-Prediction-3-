import os
import sys
import importlib
from unittest import mock

import numpy as np
import pytest


# -------------------------------------------------------------------
# Test doubles / fixtures
# -------------------------------------------------------------------

class DummyModel:
    """Predictable model used so unit tests do not depend on model.pkl internals."""
    def predict(self, values):
        arr = np.asarray(values, dtype=float)
        # Return a deterministic value based on the 4 input features.
        return np.array([arr[0].sum() * 1000.0])


@pytest.fixture(scope="session")
def app_module():
    """
    Import app.py while replacing pickle.load() so tests are stable even if the
    serialized scikit-learn model was created with an older library version.
    """
    with mock.patch("pickle.load", return_value=DummyModel()):
        if "app" in sys.modules:
            del sys.modules["app"]
        module = importlib.import_module("app")

    module.app.config.update(TESTING=True)
    module.model = DummyModel()
    return module


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def valid_form(**overrides):
    data = {
        "bedrooms": "3",
        "bathrooms": "2",
        "floors": "1",
        "yr_built": "2005",
    }
    data.update(overrides)
    return data


# ===================================================================
# PASSING TESTS (1-35)
# ===================================================================

def test_01_flask_app_object_exists(app_module):
    assert app_module.app is not None


def test_02_testing_mode_enabled(app_module):
    assert app_module.app.config["TESTING"] is True


def test_03_root_route_exists(app_module):
    rules = [rule.rule for rule in app_module.app.url_map.iter_rules()]
    assert "/" in rules


def test_04_predict_route_exists(app_module):
    rules = [rule.rule for rule in app_module.app.url_map.iter_rules()]
    assert "/predict" in rules


def test_05_root_get_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_06_root_response_is_html(client):
    response = client.get("/")
    assert "text/html" in response.content_type


def test_07_root_response_not_empty(client):
    response = client.get("/")
    assert len(response.data) > 0


def test_08_root_head_is_supported(client):
    response = client.head("/")
    assert response.status_code == 200


def test_09_root_post_is_not_allowed(client):
    response = client.post("/")
    assert response.status_code == 405


def test_10_unknown_route_returns_404(client):
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404


def test_11_predict_post_valid_input_returns_200(client):
    response = client.post("/predict", data=valid_form())
    assert response.status_code == 200


def test_12_predict_response_is_html(client):
    response = client.post("/predict", data=valid_form())
    assert "text/html" in response.content_type


def test_13_predict_response_not_empty(client):
    response = client.post("/predict", data=valid_form())
    assert len(response.data) > 0


def test_14_predict_accepts_integer_strings(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="4", bathrooms="3", floors="2", yr_built="2010"
    ))
    assert response.status_code == 200


def test_15_predict_accepts_decimal_strings(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="3.5", bathrooms="2.5", floors="1.5", yr_built="2001.5"
    ))
    assert response.status_code == 200


def test_16_predict_accepts_zero_values(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="0", bathrooms="0", floors="0", yr_built="0"
    ))
    assert response.status_code == 200


def test_17_predict_accepts_negative_numeric_values(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="-1", bathrooms="-2", floors="-1", yr_built="-2000"
    ))
    assert response.status_code == 200


def test_18_predict_accepts_scientific_notation(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="3e0", bathrooms="2e0", floors="1e0", yr_built="2.005e3"
    ))
    assert response.status_code == 200


def test_19_predict_get_without_form_returns_400(client):
    response = client.get("/predict")
    assert response.status_code == 400


def test_20_missing_bedrooms_returns_400(client):
    data = valid_form()
    data.pop("bedrooms")
    response = client.post("/predict", data=data)
    assert response.status_code == 400


def test_21_missing_bathrooms_returns_400(client):
    data = valid_form()
    data.pop("bathrooms")
    response = client.post("/predict", data=data)
    assert response.status_code == 400


def test_22_missing_floors_returns_400(client):
    data = valid_form()
    data.pop("floors")
    response = client.post("/predict", data=data)
    assert response.status_code == 400


def test_23_missing_yr_built_returns_400(client):
    data = valid_form()
    data.pop("yr_built")
    response = client.post("/predict", data=data)
    assert response.status_code == 400


def test_24_model_has_predict_method(app_module):
    assert hasattr(app_module.model, "predict")


def test_25_dummy_model_prediction_is_numpy_array():
    model = DummyModel()
    prediction = model.predict([[3, 2, 1, 2005]])
    assert isinstance(prediction, np.ndarray)


def test_26_dummy_model_returns_single_prediction():
    model = DummyModel()
    prediction = model.predict([[3, 2, 1, 2005]])
    assert prediction.shape == (1,)


def test_27_dummy_model_prediction_is_positive_for_valid_house():
    model = DummyModel()
    prediction = model.predict([[3, 2, 1, 2005]])
    assert prediction[0] > 0


def test_28_form_contains_exact_four_expected_features():
    assert set(valid_form().keys()) == {
        "bedrooms", "bathrooms", "floors", "yr_built"
    }


def test_29_numeric_features_can_convert_to_float64():
    values = np.array(list(valid_form().values())).astype(np.float64)
    assert values.dtype == np.float64


def test_30_numeric_feature_array_has_four_values():
    values = np.array(list(valid_form().values())).astype(np.float64)
    assert values.shape == (4,)


def test_31_large_numeric_values_are_accepted(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="1000", bathrooms="1000", floors="100", yr_built="9999"
    ))
    assert response.status_code == 200


def test_32_whitespace_numeric_values_are_accepted(client):
    response = client.post("/predict", data=valid_form(
        bedrooms=" 3 ", bathrooms=" 2 ", floors=" 1 ", yr_built=" 2005 "
    ))
    assert response.status_code == 200


def test_33_plus_sign_numeric_values_are_accepted(client):
    response = client.post("/predict", data=valid_form(
        bedrooms="+3", bathrooms="+2", floors="+1", yr_built="+2005"
    ))
    assert response.status_code == 200


def test_34_predict_route_supports_post_method(app_module):
    predict_rule = next(
        rule for rule in app_module.app.url_map.iter_rules()
        if rule.rule == "/predict"
    )
    assert "POST" in predict_rule.methods


def test_35_predict_route_supports_get_method(app_module):
    predict_rule = next(
        rule for rule in app_module.app.url_map.iter_rules()
        if rule.rule == "/predict"
    )
    assert "GET" in predict_rule.methods


# ===================================================================
# INTENTIONALLY FAILING TESTS (36-45)
# These are deliberately incorrect expectations for CI training/demo.
# ===================================================================

def test_36_FAIL_root_should_return_201(client):
    response = client.get("/")
    assert response.status_code == 201


def test_37_FAIL_predict_should_return_json(client):
    response = client.post("/predict", data=valid_form())
    assert response.is_json is True


def test_38_FAIL_unknown_route_should_return_200(client):
    response = client.get("/unknown")
    assert response.status_code == 200


def test_39_FAIL_root_post_should_be_allowed(client):
    response = client.post("/")
    assert response.status_code == 200


def test_40_FAIL_predict_get_should_succeed_without_form(client):
    response = client.get("/predict")
    assert response.status_code == 200


def test_41_FAIL_missing_bedrooms_should_succeed(client):
    data = valid_form()
    data.pop("bedrooms")
    response = client.post("/predict", data=data)
    assert response.status_code == 200


def test_42_FAIL_app_should_have_health_route(app_module):
    rules = [rule.rule for rule in app_module.app.url_map.iter_rules()]
    assert "/health" in rules


def test_43_FAIL_prediction_should_be_string():
    prediction = DummyModel().predict([[3, 2, 1, 2005]])
    assert isinstance(prediction, str)


def test_44_FAIL_feature_count_should_be_five():
    values = np.array(list(valid_form().values()))
    assert len(values) == 5


def test_45_FAIL_model_prediction_should_be_negative():
    prediction = DummyModel().predict([[3, 2, 1, 2005]])
    assert prediction[0] < 0


# ===================================================================
# SKIPPED TESTS (46-50)
# ===================================================================

@pytest.mark.skip(reason="Requires a live Azure App Service deployment")
def test_46_SKIP_live_app_service_health_check():
    assert True


@pytest.mark.skip(reason="Requires production database/integration environment")
def test_47_SKIP_database_integration():
    assert True


@pytest.mark.skip(reason="Performance/load testing is outside unit-test scope")
def test_48_SKIP_concurrent_1000_users():
    assert True


@pytest.mark.skip(reason="Requires browser automation such as Selenium/Playwright")
def test_49_SKIP_browser_ui_validation():
    assert True


@pytest.mark.skip(reason="Requires a production-trained model quality threshold")
def test_50_SKIP_production_model_accuracy_threshold():
    assert True
