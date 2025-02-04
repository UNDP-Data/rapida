import pytest
from unittest.mock import patch
from cbsurge.exposure.population.pop_coefficient import get_pop_coefficient_world_bank


def mock_world_bank_api(*args, **kwargs):
    class MockResponse:
        status_code = 200

        @staticmethod
        def json():
            return [
              {
                "page": 1,
                "pages": 1,
                "per_page": 50,
                "total": 4,
                "sourceid": "2",
                "lastupdated": "2025-01-28"
              },
              [
                {
                  "indicator": {
                    "id": "SP.POP.TOTL",
                    "value": "Population, total"
                  },
                  "country": {
                    "id": "KE",
                    "value": "Kenya"
                  },
                  "countryiso3code": "KEN",
                  "date": "2023",
                  "value": 55339003,
                  "unit": "",
                  "obs_status": "",
                  "decimal": 0
                },
                {
                  "indicator": {
                    "id": "SP.POP.TOTL",
                    "value": "Population, total"
                  },
                  "country": {
                    "id": "KE",
                    "value": "Kenya"
                  },
                  "countryiso3code": "KEN",
                  "date": "2022",
                  "value": 54252461,
                  "unit": "",
                  "obs_status": "",
                  "decimal": 0
                },
                {
                  "indicator": {
                    "id": "SP.POP.TOTL",
                    "value": "Population, total"
                  },
                  "country": {
                    "id": "KE",
                    "value": "Kenya"
                  },
                  "countryiso3code": "KEN",
                  "date": "2021",
                  "value": 53219166,
                  "unit": "",
                  "obs_status": "",
                  "decimal": 0
                },
                {
                  "indicator": {
                    "id": "SP.POP.TOTL",
                    "value": "Population, total"
                  },
                  "country": {
                    "id": "KE",
                    "value": "Kenya"
                  },
                  "countryiso3code": "KEN",
                  "date": "2020",
                  "value": 52217334,
                  "unit": "",
                  "obs_status": "",
                  "decimal": 0
                }
              ]
            ]


    return MockResponse()


def mock_world_bank_api_no_data(*args, **kwargs):
    class MockResponse:
        status_code = 200

        @staticmethod
        def json():
            return [{"page": 1, "pages": 1, "per_page": 50, "total": 0}, []]

    return MockResponse()


@patch('requests.get', side_effect=mock_world_bank_api)
def test_get_pop_coefficient_world_bank(mock_get):
    target_year = 2020
    country_code = "KEN"
    latest_year, coefficient = get_pop_coefficient_world_bank(target_year, country_code)

    assert latest_year == 2023, "Latest year should be 2023"
    assert coefficient == 55339003 / 52217334, "Coefficient calculation is incorrect"


@patch('requests.get', side_effect=mock_world_bank_api)
def test_get_pop_coefficient_world_bank_incorrect_target_year(mock_get):
    target_year = 2024
    country_code = "KEN"
    with pytest.raises(RuntimeError, match="No population data available for 2024."):
        get_pop_coefficient_world_bank(target_year, country_code)


@patch('requests.get', side_effect=mock_world_bank_api_no_data)
def test_get_pop_coefficient_world_bank_no_data(mock_get):
    target_year = 2020
    country_code = "KENN"
    with pytest.raises(RuntimeError, match="No population data available."):
        get_pop_coefficient_world_bank(target_year, country_code)

