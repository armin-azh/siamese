class Worker:
    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._stream = []
        super(Worker, self).__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    def add_stream_by_name(self, n_source):
        """
        add new source to the list
        :param n_source:
        :return:
        """
        self._stream.append(n_source)

    def get_stream_by_name(self, n: str):
        """
        get stream source by the given name
        :param n: name
        :return: None if such source name is not exits or the source
        """
        _s = None
        for st in self._stream:
            if n == st.name:
                _s = st
                break
        return _s

    def _run(self):
        raise NotImplementedError

    def exec_(self):
        self._run()
