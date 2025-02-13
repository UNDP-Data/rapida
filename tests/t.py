import json
from cbsurge.components.population.variables import generate_variables

print(json.dumps(generate_variables(), indent=2))