# -*- coding: utf-8 -*-
# @Time    : 6/30/21 5:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest
import coverage
# import argparse
#
# parser = argparse.ArgumentParser(description="Test runner")
# parser.add_argument('-c', '--coverage', action='store_true')
# test_args = parser.parse_args()

COV_FLAG = True
if __name__ == "__main__":
    if COV_FLAG:
        cov = coverage.coverage()
        cov.start()

    loader = unittest.TestLoader()
    tests = loader.discover('.', pattern='test_*.py')
    testRunner = unittest.runner.TextTestRunner()
    test_results = testRunner.run(tests)

    if COV_FLAG:
        cov.stop()
        cov.save()
        # 命令行模式展示结果
        cov.report()
        # 生成HTML覆盖率报告
        cov.html_report(directory='./tests/covhtml')
        # cov.xml_report(outfile='./tests/cov_report.xml')


    if test_results.wasSuccessful():  # used in github actions to make sure actions fail when tests fails
        exit(0)
    else:
        exit(1)
