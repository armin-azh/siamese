from .exceptions import *

class BaseSerializer:

    def __init__(self,name:None,*args,**kwargs) -> None:
        self._name = self.__class__.__name__ if name is None else name
        self._data = None
        self._is_checked = False
        super(BaseSerializer).__init__(*args,**kwargs)

    def __valid_seri(self)->None:
        try:
            if not self.field:
                raise EmptyFieldError("No field had been regestered")
        except Exception:
            raise EmptyFieldError("No field had been regestered")
            

    def validate(self)->bool:
        raise NotImplementedError
    
    @property
    def cleaned_data(self):
        raise NotImplementedError
    

    def __call__(self,data,many:bool=False):
        """
        create a list of instances if bool is True else create a single value
        """
        self.__valid_seri()
        if many:
            self._data = list()
            for ins in data:
                for key in ins.keys():
                    _tm = dict()
                    _tm[key] = self._field[key](ins[key])
                self._data.append(_tm)
        else:
            self._data = dict()
            for key in data.keys():
                _tm = dict()
                _tm[key] = self._field[key](data[key])
                

