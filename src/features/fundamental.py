"""Fundamental analysis feature extraction"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FundamentalAnalyzer:
    """
    Extract fundamental features from financial statements

    Features:
    - Profitability ratios (ROE, ROA, Profit Margin)
    - Liquidity ratios (Current, Quick)
    - Leverage ratios (Debt-to-Equity, Interest Coverage)
    - Valuation ratios (P/E, P/B, P/S)
    - Growth metrics (Revenue, Earnings, EPS growth)
    - Efficiency ratios (Asset Turnover, Inventory Turnover)
    """

    def __init__(self, lookback_periods: int = 4):
        """
        Initialize fundamental analyzer

        Args:
            lookback_periods: Number of quarters/years to look back for growth
        """
        self.lookback_periods = lookback_periods

    def compute_all(
        self,
        financials: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute all fundamental ratios

        Args:
            financials: Dictionary with keys:
                - "income": Income statement DataFrame
                - "balance": Balance sheet DataFrame
                - "cashflow": Cash flow statement DataFrame
            market_data: DataFrame with market price and shares outstanding

        Returns:
            DataFrame with fundamental ratios by period
        """
        logger.info("Computing fundamental ratios")

        # Extract statements
        income = financials.get("income")
        balance = financials.get("balance")
        cashflow = financials.get("cashflow")

        if income is None or balance is None:
            raise ValueError("Income statement and balance sheet are required")

        # Compute ratio categories
        profitability = self._compute_profitability(income, balance)
        liquidity = self._compute_liquidity(balance)
        leverage = self._compute_leverage(income, balance)
        efficiency = self._compute_efficiency(income, balance)
        growth = self._compute_growth(income)

        # Combine all ratios
        ratios = pd.concat(
            [profitability, liquidity, leverage, efficiency, growth],
            axis=1,
        )

        # Add valuation ratios if market data provided
        if market_data is not None:
            valuation = self._compute_valuation(income, balance, market_data)
            ratios = pd.concat([ratios, valuation], axis=1)

        logger.info(f"Computed {len(ratios.columns)} fundamental ratios")

        return ratios

    def _compute_profitability(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute profitability ratios"""
        ratios = pd.DataFrame(index=income.index)

        # ROE (Return on Equity)
        net_income = income.get("net_income", income.get("netIncome", 0))
        equity = balance.get("total_equity", balance.get("totalStockholderEquity", 0))
        ratios["roe"] = (net_income / equity.replace(0, np.nan)) * 100

        # ROA (Return on Assets)
        total_assets = balance.get("total_assets", balance.get("totalAssets", 0))
        ratios["roa"] = (net_income / total_assets.replace(0, np.nan)) * 100

        # Profit Margin
        revenue = income.get("revenue", income.get("totalRevenue", 0))
        ratios["profit_margin"] = (net_income / revenue.replace(0, np.nan)) * 100

        # Gross Profit Margin
        gross_profit = income.get("gross_profit", income.get("grossProfit", 0))
        ratios["gross_margin"] = (gross_profit / revenue.replace(0, np.nan)) * 100

        # Operating Margin
        operating_income = income.get("operating_income", income.get("operatingIncome", 0))
        ratios["operating_margin"] = (operating_income / revenue.replace(0, np.nan)) * 100

        return ratios

    def _compute_liquidity(self, balance: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidity ratios"""
        ratios = pd.DataFrame(index=balance.index)

        # Current Ratio
        current_assets = balance.get("current_assets", balance.get("totalCurrentAssets", 0))
        current_liabilities = balance.get(
            "current_liabilities", balance.get("totalCurrentLiabilities", 0)
        )
        ratios["current_ratio"] = current_assets / current_liabilities.replace(0, np.nan)

        # Quick Ratio (Acid Test)
        inventory = balance.get("inventory", balance.get("inventory", 0))
        quick_assets = current_assets - inventory
        ratios["quick_ratio"] = quick_assets / current_liabilities.replace(0, np.nan)

        # Cash Ratio
        cash = balance.get("cash", balance.get("cash", 0))
        ratios["cash_ratio"] = cash / current_liabilities.replace(0, np.nan)

        return ratios

    def _compute_leverage(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute leverage ratios"""
        ratios = pd.DataFrame(index=balance.index)

        # Debt-to-Equity
        total_debt = balance.get("total_debt", balance.get("totalDebt", 0))
        equity = balance.get("total_equity", balance.get("totalStockholderEquity", 0))
        ratios["debt_to_equity"] = total_debt / equity.replace(0, np.nan)

        # Debt-to-Assets
        total_assets = balance.get("total_assets", balance.get("totalAssets", 0))
        ratios["debt_to_assets"] = total_debt / total_assets.replace(0, np.nan)

        # Interest Coverage
        operating_income = income.get("operating_income", income.get("operatingIncome", 0))
        interest_expense = income.get("interest_expense", income.get("interestExpense", 1))
        ratios["interest_coverage"] = operating_income / interest_expense.replace(0, np.nan)

        # Equity Multiplier
        ratios["equity_multiplier"] = total_assets / equity.replace(0, np.nan)

        return ratios

    def _compute_efficiency(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute efficiency ratios"""
        ratios = pd.DataFrame(index=income.index)

        # Asset Turnover
        revenue = income.get("revenue", income.get("totalRevenue", 0))
        total_assets = balance.get("total_assets", balance.get("totalAssets", 1))
        ratios["asset_turnover"] = revenue / total_assets.replace(0, np.nan)

        # Inventory Turnover
        cogs = income.get("cost_of_goods_sold", income.get("costOfRevenue", 0))
        inventory = balance.get("inventory", balance.get("inventory", 1))
        ratios["inventory_turnover"] = cogs / inventory.replace(0, np.nan)

        # Receivables Turnover
        receivables = balance.get("accounts_receivable", balance.get("netReceivables", 1))
        ratios["receivables_turnover"] = revenue / receivables.replace(0, np.nan)

        return ratios

    def _compute_growth(self, income: pd.DataFrame) -> pd.DataFrame:
        """Compute growth metrics"""
        ratios = pd.DataFrame(index=income.index)

        # Revenue Growth (YoY)
        revenue = income.get("revenue", income.get("totalRevenue", 0))
        ratios["revenue_growth"] = revenue.pct_change(periods=4) * 100  # 4 quarters = YoY

        # Earnings Growth (YoY)
        net_income = income.get("net_income", income.get("netIncome", 0))
        ratios["earnings_growth"] = net_income.pct_change(periods=4) * 100

        # EPS Growth (YoY)
        eps = income.get("eps", income.get("eps", 0))
        ratios["eps_growth"] = eps.pct_change(periods=4) * 100

        return ratios

    def _compute_valuation(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute valuation ratios"""
        ratios = pd.DataFrame(index=income.index)

        # Get market cap (price * shares outstanding)
        price = market_data.get("price", market_data.get("close", 0))
        shares = market_data.get("shares_outstanding", 1)
        market_cap = price * shares

        # P/E Ratio
        net_income = income.get("net_income", income.get("netIncome", 1))
        ratios["pe_ratio"] = market_cap / net_income.replace(0, np.nan)

        # P/B Ratio
        equity = balance.get("total_equity", balance.get("totalStockholderEquity", 1))
        ratios["pb_ratio"] = market_cap / equity.replace(0, np.nan)

        # P/S Ratio
        revenue = income.get("revenue", income.get("totalRevenue", 1))
        ratios["ps_ratio"] = market_cap / revenue.replace(0, np.nan)

        # EV/EBITDA
        total_debt = balance.get("total_debt", balance.get("totalDebt", 0))
        cash = balance.get("cash", balance.get("cash", 0))
        enterprise_value = market_cap + total_debt - cash

        ebitda = income.get("ebitda", income.get("ebitda", 1))
        ratios["ev_ebitda"] = enterprise_value / ebitda.replace(0, np.nan)

        return ratios

    def normalize_ratios(
        self,
        ratios: pd.DataFrame,
        method: str = "z_score",
    ) -> pd.DataFrame:
        """
        Normalize ratios for ML models

        Args:
            ratios: DataFrame with fundamental ratios
            method: Normalization method ("z_score", "min_max", "robust")

        Returns:
            Normalized DataFrame
        """
        if method == "z_score":
            # Z-score normalization
            normalized = (ratios - ratios.mean()) / ratios.std()

        elif method == "min_max":
            # Min-max scaling to [0, 1]
            normalized = (ratios - ratios.min()) / (ratios.max() - ratios.min())

        elif method == "robust":
            # Robust scaling using median and IQR
            median = ratios.median()
            q75 = ratios.quantile(0.75)
            q25 = ratios.quantile(0.25)
            iqr = q75 - q25
            normalized = (ratios - median) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Replace inf and -inf with NaN
        normalized = normalized.replace([np.inf, -np.inf], np.nan)

        return normalized


class FinancialDataParser:
    """
    Parse financial statements from various sources

    Supports:
    - SEC EDGAR filings (10-K, 10-Q)
    - Yahoo Finance
    - Alpha Vantage
    - Manual CSV files
    """

    @staticmethod
    def parse_sec_filing(filing_text: str) -> Dict[str, pd.DataFrame]:
        """
        Parse SEC filing (10-K or 10-Q) text

        Args:
            filing_text: Raw filing text

        Returns:
            Dictionary with parsed financial statements
        """
        # This is a placeholder - actual implementation would parse XBRL or HTML
        # For production, use libraries like sec-edgar-downloader or sec-api
        logger.warning("SEC filing parsing not fully implemented - use dedicated library")

        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
        }

    @staticmethod
    def parse_csv(
        income_path: str,
        balance_path: str,
        cashflow_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Parse financial statements from CSV files

        Args:
            income_path: Path to income statement CSV
            balance_path: Path to balance sheet CSV
            cashflow_path: Path to cash flow statement CSV (optional)

        Returns:
            Dictionary with parsed DataFrames
        """
        statements = {
            "income": pd.read_csv(income_path, index_col=0, parse_dates=True),
            "balance": pd.read_csv(balance_path, index_col=0, parse_dates=True),
        }

        if cashflow_path:
            statements["cashflow"] = pd.read_csv(cashflow_path, index_col=0, parse_dates=True)

        return statements


# Example usage
if __name__ == "__main__":
    # Create sample financial data
    dates = pd.date_range("2020-01-01", periods=16, freq="Q")  # Quarterly data

    income = pd.DataFrame(
        {
            "revenue": np.random.randint(10000, 15000, len(dates)),
            "net_income": np.random.randint(1000, 2000, len(dates)),
            "gross_profit": np.random.randint(5000, 7000, len(dates)),
            "operating_income": np.random.randint(2000, 3000, len(dates)),
            "eps": np.random.uniform(1.0, 2.0, len(dates)),
        },
        index=dates,
    )

    balance = pd.DataFrame(
        {
            "total_assets": np.random.randint(50000, 60000, len(dates)),
            "total_equity": np.random.randint(20000, 25000, len(dates)),
            "total_debt": np.random.randint(15000, 20000, len(dates)),
            "current_assets": np.random.randint(15000, 20000, len(dates)),
            "current_liabilities": np.random.randint(8000, 10000, len(dates)),
            "cash": np.random.randint(5000, 8000, len(dates)),
            "inventory": np.random.randint(2000, 3000, len(dates)),
        },
        index=dates,
    )

    # Compute ratios
    analyzer = FundamentalAnalyzer()
    ratios = analyzer.compute_all({"income": income, "balance": balance})

    print("Fundamental Ratios:")
    print(ratios.tail())
    print(f"\nTotal ratios: {len(ratios.columns)}")
    print(f"Columns: {ratios.columns.tolist()}")
