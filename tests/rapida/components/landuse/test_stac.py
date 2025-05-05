import pytest
from datetime import date
from rapida.components.landuse.stac import create_date_range
from unittest.mock import patch

@pytest.fixture
def mock_today():
    with patch("rapida.components.landuse.stac.date") as mock_date:
        mock_date.today.return_value = date(2025, 5, 2)
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        yield

def test_default_current_year(mock_today):
    assert create_date_range(None) == "2024-11-02/2025-05-02"

def test_future_year_becomes_current(mock_today):
    assert create_date_range(2026) == "2024-11-02/2025-05-02"

def test_past_year_no_month(mock_today):
    assert create_date_range(2024) == "2024-06-30/2024-12-31"

def test_past_year_with_month(mock_today):
    assert create_date_range(2024, 10) == "2024-04-30/2024-10-31"

def test_future_month_clamped_to_today(mock_today):
    assert create_date_range(2025, 12) == "2024-11-02/2025-05-02"

def test_same_month_as_today(mock_today):
    assert create_date_range(2025, 5) == "2024-11-02/2025-05-02"

def test_past_month_in_current_year(mock_today):
    assert create_date_range(2025, 3) == "2024-09-30/2025-03-31"

def test_past_month_in_current_year_with_duration(mock_today):
    assert create_date_range(2025, 3, 3) == "2024-12-31/2025-03-31"