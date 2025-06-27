import requests
import logging
import numpy as np

from rapida.util.countries import COUNTRY_CODES



logger = logging.getLogger(__name__)


def fetch_population(country_code=None):

    """Fetch population data from World Bank API for a given country and year range."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SP.POP.TOTL?format=json"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()

    if not isinstance(data, list) or len(data) < 2:
        raise Exception("Invalid response format")

    records = data[1]

    # Extract year and population
    population_data = {
        int(entry["date"]): entry["value"]
        for entry in records if entry["value"] is not None
    }

    return population_data


def get_population_linear(year:int=None, country_code:str=None):
    """Predict future population using NumPy least squares regression."""
    assert country_code in COUNTRY_CODES, f'Invalid country_code={country_code}'

    population_data = fetch_population(country_code=country_code)
    years, pop = zip(*population_data.items())
    min_year = min(years)

    assert min_year<=year, f'year={year} is invalid . Needs to be ilarger than {min_year}'
    if year in population_data:
        return population_data[year]

    # Fit a linear model: y = a * x + b
    A = np.vstack([years, np.ones_like(years)]).T  # Design matrix
    a, b = np.linalg.lstsq(A, pop, rcond=None)[0]  # Solve least squares

    # Predict future values
    return int(a * year + b)

def get_population(year: int, country_code: str = None):
    """Estimate population for a given year using dynamic growth rate."""
    assert country_code in COUNTRY_CODES, f'Invalid country_code={country_code}'

    # Fetch population data
    population_data = fetch_population(country_code=country_code)
    years, pop = zip(*population_data.items())
    years = np.array(years)
    pop = np.array(pop)

    min_year = min(years)
    assert min_year <= year, f'Year={year} is invalid. Needs to be larger than {min_year}'

    # If the year is already in the data, return the population
    if year in population_data:
        return population_data[year]

    # Estimate growth rate dynamically
    growth_rate = estimate_growth_rate(years, pop)

    # Find the most recent data point
    latest_year = years[-1]
    latest_population = pop[-1]

    # Extrapolate the population using the estimated growth rate
    predicted_population = latest_population * (1 + growth_rate) ** (year - latest_year)

    return int(predicted_population)

def estimate_growth_rate(years, populations):
    """Estimate the annual growth rate using linear regression on log-transformed data."""
    years = np.array(years, dtype=np.float64)
    populations = np.array(populations, dtype=np.float64)

    if len(years) < 2:
        raise ValueError("At least two data points are required to estimate the growth rate.")

    # Avoid log(0) issues
    if np.any(populations <= 0):
        raise ValueError("Population values must be positive for log transformation.")

    log_population = np.log(populations)

    # Fit a linear model to log-transformed population data
    slope, _ = np.polyfit(years, log_population, 1)  # y = slope * x + intercept

    return slope  # This is equivalent to the estimated annual growth rate

def get_pop_coeff(base_year=None, target_year=None, country_code=None):
    """
    Compute the coeff between population of a country in two different years
    :param base_year: int, tha reference year, 2020 for WPOP gridded data
    :param target_year: the year we wish to compute the coeff for
    :param country_code: str, ISO3 country code
    :return: float, the coeff.
    This is larger than 1 if the target year pop is larger than base year pop
    and small if opposite is true
    """
    base_year_pop = get_population(year=base_year, country_code=country_code)
    target_year_pop = get_population(year=target_year, country_code=country_code)
    return target_year_pop/base_year_pop




if __name__ == '__main__':
    # latest_year, value = get_pop_coefficient(2020, "KEN")
    #
    # print(latest_year, value )
    country_code = 'MDA'
    coeff = get_pop_coeff(target_year=2024, country_code=country_code)
    print(coeff)
    print(get_population(year=2020, country_code=country_code))
    print(get_population(year=2040, country_code=country_code))

