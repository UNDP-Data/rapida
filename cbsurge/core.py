import json
import re

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
    file_patterns: typing.List[str]
    dependencies: typing.Optional[typing.Any] = None
    extractor:str = r"\{([^}]+)\}"


    def source_files(self, **kwargs):
        files = []
        for patt in self.file_patterns:
            variables = set(re.findall(self.extractor, patt))
            for varname in variables:
                assert varname in kwargs, f'"{varname}" kwarg is required to generate source files'
            files.append(patt.format(**kwargs))
        return files




if __name__ == '__main__':

    #list_surge_packages()
    with open('./exposure/population/variables.json') as vsrc:
        json_data = vsrc.read()
        v = Variable.model_validate_json(json_data, strict=False)
        print(v)
        s = v.source_files(year=2020, iso3_country_code = 'ALB')
        print(s)
