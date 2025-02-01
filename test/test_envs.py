import unittest

import torch

class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_cuda(self):
        self.assertTrue(torch.cuda.is_available(),msg="cuda 不可用")



if __name__ == '__main__':
    unittest.main()
