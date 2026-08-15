from AlgorithmImports import *


class CatalystMirror(QCAlgorithm):
    """Mirrors the repo strategy's universe and cadence inside LEAN.

    LEAN contributes what the native engine does not model: corporate actions
    (splits), early assignment/exercise, dividends and margin. Trade selection
    is intentionally the same so a divergence points at MODELLING rather than
    at a different set of decisions.
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 2)
        self.SetEndDate(2024, 2, 15)
        self.SetCash(100000)
        for ticker in ['SPY']:
            eq = self.AddEquity(ticker, Resolution.Minute)
            opt = self.AddOption(ticker, Resolution.Minute)
            opt.SetFilter(-10, 10, timedelta(23), timedelta(47))
