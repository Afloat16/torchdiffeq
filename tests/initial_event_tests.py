import unittest

import torch

from torchdiffeq import odeint
from problems import FIXED_METHODS


class TestInitialFixedGridEvents(unittest.TestCase):
    def test_initial_event_returns_without_stepping(self):
        for method in FIXED_METHODS:
            for reverse in (False, True):
                for vector_event in (False, True):
                    with self.subTest(method=method, reverse=reverse, vector_event=vector_event):
                        t = torch.tensor([1., 0. if reverse else 2.], dtype=torch.float64)
                        y0 = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
                        calls = []

                        def func(t, y):
                            calls.append(t)
                            raise AssertionError("An initial event must not evaluate the vector field.")

                        def event_fn(t, y):
                            value = y[0] - 2.
                            return torch.stack((value, y[1] + 1)) if vector_event else value

                        event_time, solution = odeint(func, y0, t, method=method,
                            options={"step_size": .1}, event_fn=event_fn)
                        torch.testing.assert_close(event_time, t[0], atol=0, rtol=0)
                        torch.testing.assert_close(solution, y0.expand(2, 2))
                        self.assertEqual(calls, [])
                        torch.testing.assert_close(torch.autograd.grad(solution[-1].sum(), y0)[0],
                                                   torch.ones_like(y0))

    def test_noninitial_event_remains_supported(self):
        for method in ("euler", "rk4", "explicit_adams", "implicit_adams"):
            with self.subTest(method=method):
                t = torch.tensor([0., 1.], dtype=torch.float64)
                event_time, solution = odeint(lambda t, y: torch.ones_like(y), t[0], t,
                    method=method, options={"step_size": .1}, event_fn=lambda t, y: y - .45)
                torch.testing.assert_close(event_time, t.new_tensor(.45), atol=1e-8, rtol=1e-8)
                torch.testing.assert_close(solution[-1], t.new_tensor(.45), atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
