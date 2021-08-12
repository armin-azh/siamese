from typing import Any

from .exceptions import *


class BaseField:
    def __init__(self,name:str,dtype,required:bool,*args,**kwargs) -> None:
        self._dtype = dtype
        self._name = name
        self._required = required
        self._value = None
        super(BaseField).__init__(*args,**kwargs)

    def validate(self):
        raise NotImplementedError
    
    @property
    def cleaned_date(self):
        raise NotImplementedError

    def __call__(self,val):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._dtype

    def __check_dtype(self)->bool:
        """
        check the value is an instance of a specific data type
        """
        if self._value is None:
            raise SetValueError("no value had been determined")
        return True if isinstance(self._value,self._dtype) else False



class CharField(BaseField):
    def __init__(self, name: str,required: bool, *args, **kwargs) -> None:
        super().__init__(name, str, required, *args, **kwargs)

    def validate(self)->bool:
        return self.__check_dtype()

    def __call__(self, val:str)->None:
        self._value = val

    @property
    def cleaned_date(self)->str:
        return self._value
