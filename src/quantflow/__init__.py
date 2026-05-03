from .core import calculate_returns, infer_frequency
from .core import portfolio_core
from . import load_data

# important: importing this registers df.qf
from .accessors import QFAccessor, ReturnsNamespace

# utils namespace - only expose the 3 new functions
from .core.utils import portfolio_returns, portfolio_volatility, plot_ef2
class utils:
    pass

utils.portfolio_returns = portfolio_returns
utils.portfolio_volatility = portfolio_volatility
utils.plot_ef2 = plot_ef2

# portfolio namespace
class portfolio:
    pass

portfolio.optimize_allocation_for_target_return = portfolio_core.optimize_allocation_for_target_return
portfolio.get_price_data = portfolio_core.get_price_data
portfolio.portfolio_returns = portfolio_core.portfolio_returns
portfolio.portfolio_volatility = portfolio_core.portfolio_volatility
portfolio.optimal_weights = portfolio_core.optimal_weights
portfolio.plot_ef = portfolio_core.plot_ef
portfolio.minimize_vol = portfolio_core.minimize_vol

__all__ = [
    "calculate_returns",
    "infer_frequency",
    "load_data",
    "QFAccessor",
    "ReturnsNamespace",
    "utils",
    "portfolio",
]