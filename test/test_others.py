import unittest

import torch


class MyTestCase(unittest.TestCase):
    @unittest.skip
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_nb_distribution(self):
        noise = torch.distributions.NegativeBinomial(30,0.5).sample(torch.Size((8,8)))
        self.assertIsNotNone(noise)
        print(noise)
        mean = torch.mean(noise)
        variance = torch.var(noise)
        print(f"Mean: {mean}, Variance: {variance}")


    def test_normal_distribution(self):
        noise = torch.distributions.Normal(0,1).sample(torch.Size((8, 8)))
        self.assertIsNotNone(noise)
        print(noise)

    def test_gama_distribution(self):
        noise = torch.distributions.Gamma(30, 0.001).sample(torch.Size((8, 8)))
        self.assertIsNotNone(noise)
        print(noise)

if __name__ == '__main__':
    unittest.main()
