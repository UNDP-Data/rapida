import json
from sympy.parsing.sympy_parser import parse_expr
from osgeo_utils import gdal_calc
from pydantic import BaseModel, Field
from typing import Optional, List
import pkgutil
import os
import re

from cbsurge.session import Session


def list_surge_packages(path=os.path.dirname(__file__), valid_names = ('exposure',)):
    for e in pkgutil.walk_packages([path]):
        for vname in valid_names:
            if e.name.startswith(vname):
                print(e)


class Variable(BaseModel):
    name: str
    title: str
    source: str = None
    files: Optional[List[str]] = None
    _extractor_:str = r"\{([^}]+)\}"
    _default_operators_ = '+-/*%'
    is_computable: Optional[bool] = True
    is_raster: Optional[bool] = True
    #operators: Optional[set[str]] = None
    #_source_file_:str = None

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

    def download(self, **kwargs):
        pass

    def compute(self):
        parsed_expr = parse_expr(self.source)
        variables = parsed_expr.free_symbols

    def __call__(self, *args, **kwargs):
        raise NotImplemented(f'Only subclasses of {self.__class__.__name__} implement this method')

    def resolve(self, **kwargs):
        if self.source is not None:

            operators = set(self._default_operators_).intersection(self.source)
            if operators:pass
        else:
            self.is_computable = False




if __name__ == '__main__':

    with Session() as ses:
        for var_name, var_data in ses.config['variables']['population'].items():

            v = Variable(name=var_name, **var_data)
            print(v)
