import json
from sympy.parsing.sympy_parser import parse_expr
from osgeo_utils import gdal_calc
from pydantic import BaseModel, Field
import typing
import pkgutil
import os
import re
def list_surge_packages(path=os.path.dirname(__file__), valid_names = ('exposure',)):
    for e in pkgutil.walk_packages([path]):
        for vname in valid_names:
            if e.name.startswith(vname):
                print(e)


class Variable(BaseModel):
    id: str
    name: str
    source: str = None
    file: str = None
    _extractor_:str = r"\{([^}]+)\}"
    _default_operators_ = '+-/*'
    computed: typing.Optional[bool] = None
    operators: typing.Optional[set[str]] = None
    _source_file_:str = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resolve(**kwargs)

    def resolve_file(self, **kwargs):
        if 'iso3_country_code' in kwargs:
            kwargs['iso3_country_code_lower'] = kwargs['iso3_country_code_lower'].lower()
        variables = set(re.findall(self._extractor_, self.file))
        for varname in variables:
            assert varname in kwargs, f'"{varname}" kwarg is required to generate source files'
        return self.file.format(**kwargs)

    def compute(self):
        parsed_expr = parse_expr(self.source)
        variables = parsed_expr.free_symbols



    def resolve(self, **kwargs):
        if self.source is not None:
            operators = set(self._default_operators_).intersection(self.source)
            if operators:
                self.operators = operators
                self.computed = True
                self.compute()




if __name__ == '__main__':

    #list_surge_packages()
    with open('./exposure/population/variables.json') as vsrc:
        json_data = json.load(vsrc)
        for v in json_data:
            if v['id'] == 'dependency':
                vr = Variable(**v)

                print(vr)
