import unittest

import torch

from torchdiffeq import odeint


class TestAdamsNonuniformGrid(unittest.TestCase):
    def test_polynomial_on_irregular_grid(self):
        for method in ("explicit_adams", "implicit_adams"):
            for reverse in (False, True):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(method=method, reverse=reverse, dtype=dtype):
                        t = torch.tensor([0., .1, .2, .3, .9, 1.], dtype=dtype)
                        if reverse:
                            t = t.flip(0)
                        expected = t ** 3 / 3
                        result = odeint(lambda t, y: t ** 2, expected[0], t, method=method)
                        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)

    def test_short_final_step_and_parameter_gradient(self):
        for method in ("explicit_adams", "implicit_adams"):
            with self.subTest(method=method):
                parameter = torch.tensor(2., dtype=torch.float64, requires_grad=True)
                t = torch.tensor([0., .95], dtype=torch.float64)
                result = odeint(lambda t, y: parameter * t ** 2, t[0], t,
                                method=method, options={"step_size": .1})
                expected = parameter * t ** 3 / 3
                torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
                gradient = torch.autograd.grad(result[-1], parameter)[0]
                torch.testing.assert_close(gradient, t[-1] ** 3 / 3)

    def test_regular_grid_still_uses_adams_history(self):
        from torchdiffeq._impl.fixed_adams import AdamsBashforth
        from torchdiffeq._impl.misc import _PerturbFunc

        func = _PerturbFunc(lambda t, y: t ** 2)
        func.callback_step = lambda *args: None
        solver = AdamsBashforth(func, torch.tensor(0., dtype=torch.float64))
        t = torch.linspace(0., 1., 11, dtype=torch.float64)
        result = solver.integrate(t)
        torch.testing.assert_close(result, t ** 3 / 3)
        self.assertEqual(len(solver.prev_f), 10)

    def test_fixed_step_event_path(self):
        for method in ("explicit_adams", "implicit_adams"):
            with self.subTest(method=method):
                t = torch.tensor([0., 1.], dtype=torch.float64)
                event_time, solution = odeint(lambda t, y: t ** 2, t[0], t,
                    method=method, options={"step_size": .1, "interp": "cubic"},
                    event_fn=lambda t, y: y - .125 / 3)
                torch.testing.assert_close(event_time, t.new_tensor(.5), atol=1e-8, rtol=1e-8)
                torch.testing.assert_close(solution[-1], t.new_tensor(.125 / 3))


if __name__ == "__main__":
    unittest.main()
