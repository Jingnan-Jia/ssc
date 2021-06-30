# -*- coding: utf-8 -*-
# @Time    : 6/30/21 5:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest


loader = unittest.TestLoader()
tests = loader.discover('.', pattern='test_*.py')
print(f"Testing files:{tests}")
testRunner = unittest.runner.TextTestRunner()
test_results = testRunner.run(tests)

if test_results.wasSuccessful():
    exit(0)
else:
    exit(1)
