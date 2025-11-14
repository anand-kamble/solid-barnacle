"""
Data loading utilities for journal entry data from PostgreSQL database.

This module provides functions to load journal entry, entry line, and ledger account
data from the database into pandas DataFrames.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import text

from data.psql_connection import PostgresConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)


@dataclass
class JournalEntryDataFrames:
    """Container for journal entry related DataFrames."""

    journal_entry: pd.DataFrame
    entry_line: pd.DataFrame
    ledger_account: pd.DataFrame
    journal_entry_with_lines: pd.DataFrame  # Joined journal_entry and entry_line

    def __repr__(self) -> str:
        """Return string representation with DataFrame shapes."""
        return (
            f"JournalEntryDataFrames(\n"
            f"  journal_entry: {self.journal_entry.shape},\n"
            f"  entry_line: {self.entry_line.shape},\n"
            f"  ledger_account: {self.ledger_account.shape},\n"
            f"  journal_entry_with_lines: {self.journal_entry_with_lines.shape}\n"
            f")"
        )


def load_dataframes(
    database_name: str = "",
    business_id: str = "",
    year_month: Optional[int] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    max_connections: int = 10,
    use_migration_user: bool = False,
) -> JournalEntryDataFrames:

    logger.info(f"Initializing database connection: database={database_name}, max_connections={max_connections}, use_migration_user={use_migration_user}")
    postgres = PostgresConnection(
        database_name=database_name,
        max_connections=max_connections,
        use_migration_user=use_migration_user,
    )

    engine = postgres.get_engine()
    logger.info("Database connection established")

    # Build WHERE clause and parameters for filtering (using parameterized queries)
    where_clauses = []
    params = {}

    where_clauses.append("business_id = :business_id")
    params["business_id"] = business_id
    if year_month is not None:
        where_clauses.append("year_month = :year_month")
        params["year_month"] = year_month
    if min_date is not None:
        where_clauses.append("date >= :min_date")
        params["min_date"] = min_date
    if max_date is not None:
        where_clauses.append("date <= :max_date")
        params["max_date"] = max_date

    where_clause = " AND ".join(where_clauses)
    journal_entry_where = f"WHERE {where_clause}"
    logger.info(f"Applied filters: {params}")

    # Build WHERE clause for entry_line query (with table prefix for joined query)
    # Use the same parameter names but prefix columns with 'je.'
    entry_line_where_clauses = []
    entry_line_where_clauses.append("je.business_id = :business_id")
    if year_month is not None:
        entry_line_where_clauses.append("je.year_month = :year_month")
    if min_date is not None:
        entry_line_where_clauses.append("je.date >= :min_date")
    if max_date is not None:
        entry_line_where_clauses.append("je.date <= :max_date")
    entry_line_where_clause = " AND ".join(entry_line_where_clauses)

    # Load journal_entry table
    journal_entry_query = f"""
        SELECT 
            journal_entry_id,
            business_id,
            year_month,
            date,
            number,
            description,
            currency,
            journal_entry_type,
            journal_entry_sub_type,
            journal_entry_status,
            journal_entry_origin
        FROM journal_entry
        {journal_entry_where}
        ORDER BY date, number
    """

    # Load entry_line table (filtered by journal_entry through join)
    entry_line_query = f"""
        SELECT 
            el.entry_line_id,
            el.journal_entry_id,
            el.ledger_account_id,
            el.index,
            el.description,
            el.debit,
            el.credit,
            el.exchange_rate
        FROM entry_line el
        INNER JOIN journal_entry je ON el.journal_entry_id = je.journal_entry_id
        WHERE {entry_line_where_clause}
        ORDER BY el.journal_entry_id, el.index
    """

    # Load ledger_account table (filtered by business_id)
    ledger_account_query = """
        SELECT 
            ledger_account_id,
            business_id,
            number,
            name,
            parent_ledger_account_id,
            currency,
            nature,
            ledger_account_type,
            ledger_account_sub_type,
            ledger_account_sub_sub_type,
            ledger_account_status,
            foreign_exchange_adjustment_ledger_account_id,
            cash_flow_group,
            added_date,
            removed_date,
            is_used_in_journal_entries
        FROM ledger_account
        WHERE business_id = :business_id
        ORDER BY business_id, number
    """

    with engine.connect() as conn:
        logger.info("Loading journal_entry data")
        journal_entry_df = pd.read_sql(
            text(journal_entry_query),
            conn,
            params=params,
            parse_dates=["date"],
        )
        logger.info(f"Loaded journal_entry: {journal_entry_df.shape[0]} rows")

        logger.info("Loading entry_line data")
        entry_line_df = pd.read_sql(
            text(entry_line_query),
            conn,
            params=params,
        )
        logger.info(f"Loaded entry_line: {entry_line_df.shape[0]} rows")

        logger.info("Loading ledger_account data")
        ledger_account_df = pd.read_sql(
            text(ledger_account_query),
            conn,
            params={"business_id": business_id},
            parse_dates=["added_date", "removed_date"],
        )
        logger.info(f"Loaded ledger_account: {ledger_account_df.shape[0]} rows")

    logger.info("Creating joined DataFrame: journal_entry with entry_line")
    journal_entry_with_lines_df = journal_entry_df.merge(
        entry_line_df,
        on="journal_entry_id",
        how="left",
        suffixes=("", "_entry_line"),
    )
    logger.info(f"Created journal_entry_with_lines: {journal_entry_with_lines_df.shape[0]} rows")

    # Optionally merge with ledger_account for account details
    # This could be done later if needed, but including it here for completeness
    # journal_entry_with_lines_df = journal_entry_with_lines_df.merge(
    #     ledger_account_df[['ledger_account_id', 'number', 'name', 'ledger_account_type']],
    #     on='ledger_account_id',
    #     how='left'
    # )

    result = JournalEntryDataFrames(
        journal_entry=journal_entry_df,
        entry_line=entry_line_df,
        ledger_account=ledger_account_df,
        journal_entry_with_lines=journal_entry_with_lines_df,
    )
    logger.info(f"Data loading complete: {result}")
    return result

