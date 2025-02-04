import requests
import datetime
import logging


logger = logging.getLogger(__name__)


def get_pop_coefficient(target_year: int, country_code: str) -> list[int | float]:
    """
    Fetches population data from remote sources (world bank or UNSD) and calculates the population growth coefficient.

    Compare the latest year available in both sources, and will pick the latest one to compute.

    Currently, UNSD source is not supported. It returns always data from world bank.

    Parameters:
        target_year (int): The base year for comparison.
        country_code (str): The ISO3 country code.

    Returns:
        list[int, float]: A list containing the latest available year and the calculated coefficient.
    """
    latest_year, coefficient = get_pop_coefficient_world_bank(target_year, country_code)
    return [latest_year, coefficient]


def get_pop_coefficient_world_bank(target_year: int, country_code: str) -> list[int | float]:
    """
    Fetches population data from the World Bank API and calculates the population growth coefficient.

    Parameters:
        target_year (int): The base year for comparison.
        country_code (str): The ISO3 country code.

    Returns:
        list[int, float]: A list containing the latest available year and the calculated coefficient.
    """

    current_year = datetime.datetime.now().year
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SP.POP.TOTL?format=json&date={target_year}:{current_year}"

    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    if len(data) < 2 or not isinstance(data[1], list):
        raise RuntimeError("Invalid response format or no population data available.")

    pop_data = {int(entry["date"]): entry["value"] for entry in data[1] if entry["value"] is not None}

    if target_year not in pop_data:
        raise RuntimeError(f"No population data available for {target_year}.")

    latest_year = max(pop_data.keys())

    if target_year > latest_year:
        raise RuntimeError(f"{target_year} must be earlier than the latest year {latest_year}.")

    latest_population = pop_data[latest_year]
    target_population = pop_data[target_year]

    coefficient = latest_population / target_population
    return [latest_year, coefficient]

if __name__ == '__main__':
    latest_year, value = get_pop_coefficient(2020, "KEN")

    print(latest_year, value )
