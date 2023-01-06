import unittest
from unittest.mock import MagicMock

from txrm2tiff.txrm.txrm_property import txrm_property


class ToasterError(Exception):
    pass


class Toaster:
    def __init__(self):
        self.ole = None
        self.strict = False
        self.file_is_open = True

    @txrm_property(fallback="bread")
    def toast(self):
        return self.heater()

    def heater(self):
        return "nice toast"


class TestTxrmProperty(unittest.TestCase):
    def test_txrm_property_fails_as_method(self):
        t = Toaster()
        with self.assertRaises(TypeError):
            t.toast()
            # Not a method

    def test_txrm_property_works_as_property(self):
        t = Toaster()
        self.assertEqual(t.toast, "nice toast")

    def test_delete_works(self):
        t = Toaster()
        t.heater = MagicMock(side_effect=["nice toast", "burnt toast"])
        # Check that toast's return value is cached until deleted
        self.assertEqual(t.toast, "nice toast")
        self.assertEqual(t.toast, "nice toast")
        self.assertEqual(t.toast, "nice toast")
        del t.toast
        self.assertIs(t.toast, "burnt toast")

    def test_error_returns_fallback(self):
        t = Toaster()
        t.file_is_open = True
        t.heater = MagicMock(side_effect=ToasterError("The toaster is not plugged in."))
        self.assertEqual(t.toast, "bread")

    def test_error_and_unopened_file_returns_fallback(self):
        t = Toaster()
        t.file_is_open = False
        t.heater = MagicMock(side_effect=ToasterError("The toaster is not plugged in."))
        self.assertEqual(t.toast, "bread")

    def test_strict_error_and_unopened_raises_IOError(self):
        t = Toaster()
        t.strict = True
        t.file_is_open = False
        t.heater = MagicMock(side_effect=ToasterError("The toaster is not plugged in."))
        with self.assertRaises(IOError):
            t.toast

    def test_strict_error_and_opened_raises_function_error(self):
        t = Toaster()
        t.strict = True
        t.file_is_open = True
        t.heater = MagicMock(side_effect=ToasterError("The toaster is not plugged in."))
        with self.assertRaises(ToasterError):
            t.toast

    def test_error_raised_if_strict(self):
        t = Toaster()
        t.strict = True
        t.heater = MagicMock(side_effect=ToasterError("The toaster is not plugged in."))
        with self.assertRaises(ToasterError):
            t.toast

    def test_cannot_set_attribute(self):
        t = Toaster()
        self.assertEqual(t.toast, "nice toast")
        with self.assertRaises(AttributeError):
            t.toast = "bread"
            # You cannot un-toast
