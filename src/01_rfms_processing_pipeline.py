import pandas as pd
import os
from pathlib import Path
import duckdb as db

class RFMSProcessingPipeline:
    """
    Pipeline for processing OLIST e-commerce data into RFMS (Recency, Frequency, Monetary, Satisfaction) metrics.
    """

    def __init__(self, repo_wd: Path):
        """
        Initialize the pipeline with repository working directory.

        Args:
            repo_wd: Path object pointing to the repository root directory
        """
        self.repo_wd = repo_wd
        self.orders_df = None
        self.customers_df = None
        self.payments_df = None
        self.items_df = None
        self.reviews_df = None

    def load_datasets(self) -> None:
        """Load all required datasets from raw data directory."""
        olist_orders_dataset_path = self.repo_wd / "data/raw/olist_orders_dataset.csv"
        olist_customers_dataset_path = self.repo_wd / "data/raw/olist_customers_dataset.csv"
        olist_order_payments_dataset_path = self.repo_wd / "data/raw/olist_order_payments_dataset.csv"
        olist_order_items_dataset_path = self.repo_wd / "data/raw/olist_order_items_dataset.csv"
        olist_order_reviews_dataset_path = self.repo_wd / "data/raw/olist_order_reviews_dataset.csv"

        self.orders_df = pd.read_csv(olist_orders_dataset_path)
        self.customers_df = pd.read_csv(olist_customers_dataset_path)
        self.payments_df = pd.read_csv(olist_order_payments_dataset_path)
        self.items_df = pd.read_csv(olist_order_items_dataset_path)
        self.reviews_df = pd.read_csv(olist_order_reviews_dataset_path)

    def preprocess_orders(self) -> None:
        """Convert timestamp columns and categorical columns in orders dataset."""
        timestamp_cols = [
            'order_purchase_timestamp'
        ]

        for col in timestamp_cols:
            self.orders_df[col] = pd.to_datetime(self.orders_df[col])

        self.orders_df['order_id'] = self.orders_df['order_id'].astype('category')
        self.orders_df['customer_id'] = self.orders_df['customer_id'].astype('category')
        self.orders_df['order_status'] = self.orders_df['order_status'].astype('category')

    def preprocess_customers(self) -> None:
        """Convert all columns in customers dataset to category type for memory optimization."""
        for col in self.customers_df.columns:
            self.customers_df[col] = self.customers_df[col].astype('category')

    def preprocess_items(self) -> None:
        """Convert object columns to category and timestamp columns to datetime."""
        for col in self.items_df.columns:
            if self.items_df[col].dtype == 'object':
                if col == 'shipping_limit_date':
                    self.items_df[col] = pd.to_datetime(self.items_df[col])
                else:
                    self.items_df[col] = self.items_df[col].astype('category')

    def preprocess_payments(self) -> None:
        """Convert object columns in payments dataset to category type."""
        for col in self.payments_df.columns:
            if self.payments_df[col].dtype == 'object':
                self.payments_df[col] = self.payments_df[col].astype('category')

    def preprocess_reviews(self) -> None:
        """Select relevant columns and convert order_id to category."""
        self.reviews_df = self.reviews_df[['order_id', 'review_score']]
        self.reviews_df['order_id'] = self.reviews_df['order_id'].astype('category')
        self.reviews_df = pd.DataFrame(
            self.reviews_df.groupby('order_id')['review_score'].mean().round(1)
        ).reset_index().astype('category')

    def calculate_monetary(self) -> pd.DataFrame:
        """Calculate monetary values from items dataset."""
        items_subset = self.items_df[['order_id', 'order_item_id', 'price', 'freight_value']].copy()
        items_subset['amount'] = items_subset['price'] + items_subset['freight_value']

        monetary_df = items_subset.groupby('order_id').agg(
            number_of_items=('order_item_id', 'max'),
            total_amount=('amount', 'sum')
        ).reset_index()

        return monetary_df

    def build_rfms_table(self) -> pd.DataFrame:
        """
        Build RFMS table using DuckDB for efficient processing.

        Returns:
            DataFrame containing RFMS metrics per customer
        """
        # Prepare order data with recency
        order_data = self.orders_df[['order_id', 'customer_id', 'order_purchase_timestamp']].copy()
        order_data['recency'] = (order_data['order_purchase_timestamp'].max() -
                               order_data['order_purchase_timestamp']).dt.days

        # Calculate monetary values
        monetary_df = self.calculate_monetary()

        # Aggregate payments
        payment_agg = self.payments_df.groupby('order_id')['payment_value'].sum()

        # Merge datasets
        rm_data = order_data.merge(
            pd.DataFrame(payment_agg).reset_index(),
            how='left',
            on='order_id'
        ).merge(
            monetary_df,
            how='left',
            on='order_id'
        ).merge(
            self.customers_df.reset_index(drop=True),
            how='left',
            on='customer_id'
        )

        # Fill missing payment values with total_amount
        rm_data['payment_value'] = rm_data['payment_value'].fillna(rm_data['total_amount'])
        rm_data = rm_data.drop(columns=['total_amount'])
        rm_data = rm_data.rename(columns={'payment_value': 'monetary'})

        # Merge with reviews
        rfms_data = rm_data.merge(
            self.reviews_df,
            how='left',
            on='order_id'
        )

        # Use DuckDB for aggregation
        rfms_duck_data = db.query(
            """
            WITH new_rm_data AS (
                SELECT
                    customer_unique_id,
                    order_id,
                    recency,
                    monetary,
                    review_score,
                    order_purchase_timestamp,
                    CAST(customer_zip_code_prefix AS VARCHAR) AS customer_zip_code_prefix,
                    customer_city,
                    customer_state
                FROM rfms_data
            )
            SELECT
                customer_unique_id,
                MIN(recency) AS recency,
                COUNT(customer_unique_id) AS frequency,
                SUM(monetary) AS monetary,
                ROUND(AVG(review_score), 1) AS review_score,
                MAX(order_purchase_timestamp) AS order_purchase_timestamp,
                MODE(customer_zip_code_prefix) AS customer_zip_code_prefix,
                MODE(customer_city) AS customer_city,
                MODE(customer_state) AS customer_state
            FROM new_rm_data
            GROUP BY customer_unique_id
            """
        ).df()

        rfms_duck_data['customer_zip_code_prefix'] = rfms_duck_data['customer_zip_code_prefix'].astype('category')

        return rfms_duck_data

    def split_by_review_status(self, rfms_data: pd.DataFrame) -> tuple:
        """
        Split RFMS data into active reviewers and silent customers.

        Args:
            rfms_data: DataFrame containing RFMS metrics

        Returns:
            Tuple of (active_reviewers_df, silent_customers_df)
        """
        active_reviewers = rfms_data[~rfms_data['review_score'].isna()]
        silent_customers = rfms_data[rfms_data['review_score'].isna()]

        return active_reviewers, silent_customers

    def save_processed_data(self, rfms_data: pd.DataFrame, active_reviewers: pd.DataFrame,
                           silent_customers: pd.DataFrame) -> None:
        """
        Save processed RFMS data to parquet format.

        Args:
            rfms_data: Complete RFMS dataset
            active_reviewers: Customers with reviews
            silent_customers: Customers without reviews
        """
        output_dir = self.repo_wd / "data/processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        rfms_data.to_parquet(
            output_dir / "rfms_data.parquet",
            index=False,
            engine="fastparquet"
        )

        active_reviewers.to_parquet(
            output_dir / "rfms_active_reviewers.parquet",
            index=False,
            engine="fastparquet"
        )

        silent_customers.to_parquet(
            output_dir / "rfms_silent_customers.parquet",
            index=False,
            engine="fastparquet"
        )

    def run(self) -> tuple:
        """
        Execute the complete RFMS processing pipeline.

        Returns:
            Tuple of (rfms_data, active_reviewers, silent_customers)
        """
        self.load_datasets()
        self.preprocess_orders()
        self.preprocess_customers()
        self.preprocess_items()
        self.preprocess_payments()
        self.preprocess_reviews()

        rfms_data = self.build_rfms_table()
        active_reviewers, silent_customers = self.split_by_review_status(rfms_data)

        self.save_processed_data(rfms_data, active_reviewers, silent_customers)

        return rfms_data, active_reviewers, silent_customers


if __name__ == "__main__":
    here = Path(__file__).resolve()
    repo_wd = here.parent.parent
    print(repo_wd)
    pipeline = RFMSProcessingPipeline(repo_wd)
    rfms_data, active_reviewers, silent_customers = pipeline.run()

    print("RFMS processing pipeline completed successfully.", "\n", 5 * "-----")
    print(f"RFMS Data Shape: {rfms_data.shape}")
    print(f"Active Reviewers: {len(active_reviewers)}")
    print(f"Silent Customers: {len(silent_customers)}")
