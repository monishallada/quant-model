from AlgorithmImports import *


class CatalystMirror(QCAlgorithm):
    """Runs the active strategy's RULE inside LEAN, independently.

    LEAN selects its own contracts from its own chain, prices its own fills,
    applies its own fee and margin models, and computes its own equity. Nothing
    here reads a native-engine result — only the rule is shared, which is
    unavoidable if the two engines are to be testing the same strategy.

    What LEAN contributes that the native engine cannot: corporate actions
    (splits), early assignment and exercise, dividends and margin.
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 2)
        self.SetEndDate(2024, 3, 28)
        self.SetCash(100000)
        self.moneyness = 1.05
        self.hold_days = 15
        self.entry_every = 15
        self._session = 0
        self._opened = {}
        self.symbols = []
        for ticker in ['SPY']:
            eq = self.AddEquity(ticker, Resolution.Minute)
            opt = self.AddOption(ticker, Resolution.Minute)
            opt.SetFilter(-15, 15, timedelta(23), timedelta(47))
            self.symbols.append(opt.Symbol)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(15, 45), self.Rebalance)

    def Rebalance(self):
        self._session += 1
        # Close anything past its hold, exactly as the native rule does.
        for symbol, opened in list(self._opened.items()):
            if (self._session - opened) >= self.hold_days:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
                self._opened.pop(symbol, None)
        if (self._session - 1) % self.entry_every != 0:
            return
        for canonical in self.symbols:
            chain = self.CurrentSlice.OptionChains.get(canonical)
            if chain is None:
                continue
            underlying = chain.Underlying.Price
            if underlying <= 0:
                continue
            target = underlying * self.moneyness
            calls = [c for c in chain if c.Right == OptionRight.Call and c.AskPrice > 0.05]
            if not calls:
                continue
            pick = sorted(calls, key=lambda c: (abs(c.Strike - target), c.Expiry))[0]
            if pick.Symbol in self._opened:
                continue
            # LEAN sizes against its OWN portfolio and margin model.
            qty = self.CalculateOrderQuantity(pick.Symbol, 0.05)
            if qty and qty > 0:
                self.MarketOrder(pick.Symbol, max(int(qty), 1))
                self._opened[pick.Symbol] = self._session
