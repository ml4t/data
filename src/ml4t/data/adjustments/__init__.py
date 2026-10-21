"""Corporate actions and price adjustment utilities.

The canonical convention treats a split ratio as new shares per old share on
the event date. Raw values are retained, and adjusted values use the share
basis of the latest observation.

Example:
    >>> from ml4t.data.adjustments import apply_corporate_actions
    >>> adjusted_prices = apply_corporate_actions(unadjusted_df)
    >>> adjusted_prices.select("close", "adj_close", "price_adjustment_factor")
"""

from .core import (
    apply_corporate_actions,
    apply_dividends,
    apply_splits,
)
from .history import AdjustedHistory, prepare_adjusted_history
from .inputs import apply_adjustment_factors, prepare_action_inputs

__all__ = [
    "apply_splits",
    "apply_dividends",
    "apply_corporate_actions",
    "apply_adjustment_factors",
    "prepare_action_inputs",
    "AdjustedHistory",
    "prepare_adjusted_history",
]
