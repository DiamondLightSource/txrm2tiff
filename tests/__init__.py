import unittest

def runall():
    # Includes integration tests
    suite = unittest.TestLoader().discover('.', pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)