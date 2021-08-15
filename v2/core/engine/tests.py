from unittest import TestCase
from v2.tools.logger import LOG_Path

from v2.core.engine._basic import BasicService


class BasicServiceTestCase(TestCase):
    def test_create_basic_server_object_without_name(self):
        a = BasicService(name=None, log_path=LOG_Path)
        self.assertEqual(a.name, a.__class__.__name__)

        with self.assertRaises(NotImplementedError):
            a.exec_()

    def test_create_basic_service_object_with_name(self):
        a = BasicService(name="bladeRunner", log_path=LOG_Path)
        self.assertEqual(a.name, "bladeRunner")


