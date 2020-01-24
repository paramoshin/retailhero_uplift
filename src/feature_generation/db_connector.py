from typing import Optional, List, Tuple
import sys
from pathlib import Path
p = str(Path(".").resolve().parent.parent)
sys.path.extend([p])

from sqlalchemy import func, distinct, cast, Time, extract, Integer
from sqlalchemy.sql.functions import mode
from sqlalchemy.orm import Session

from src.database.base import with_session
from src.database.database_client import DatabaseClient
from src.database.models import Clients, Products, Purchases


class DBConnector:
    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client
        self.session_factory = self.db_client.session_factory

    @with_session
    def join_clients_purchases(self, session: Session, client_ids: Optional[List[str]] = None)\
            -> Optional[List[Tuple[Clients, Purchases]]]:
        q = session.query(Clients, Purchases)
        if client_ids:
            q = q.filter(Clients.client_id.in_(client_ids))
        return q.all()

    @with_session
    def join_purchases_products(self, session: Session) -> Optional[List[Tuple[Purchases, Products]]]:
        q = session.query(Purchases, Products)
        return q.all()

    @with_session
    def generate_purchase_features(
            self, session: Session, last_month: Optional[bool] = True, client_ids: Optional[List[str]] = None
    ) -> Tuple[List, List]:
        subq_1 = session.query(
            Purchases.client_id,
            Purchases.transaction_id,
            Purchases.store_id,
            cast(Purchases.transaction_datetime, Time).label("transaction_time"),
            extract('dow', Purchases.transaction_datetime).label("transaction_weekday"),
            Purchases.regular_points_received,
            Purchases.express_points_received,
            Purchases.regular_points_spent,
            Purchases.express_points_spent,
            Purchases.purchase_sum,

            func.sum(Products.is_alcohol.cast(Integer)).label("n_alcohols"),
            func.sum(Products.is_own_trademark.cast(Integer)).label("n_own_trademark"),
            func.sum(Products.netto).label("sum_netto"),
            func.avg(Products.netto).label("avg_netto"),

            func.count(distinct(Purchases.product_id)).label("n_unique_products"),
            func.sum(Purchases.product_quantity).label("n_products"),
            (func.sum(Purchases.trn_sum_from_iss) / func.coalesce(func.nullif(func.sum(Purchases.product_quantity), 0), 1)).label("avg_product_price"),
            func.sum(Purchases.trn_sum_from_iss).label("sum_trn_sum_from_iss"),
            func.coalesce(func.sum(Purchases.trn_sum_from_red), 0).label("sum_trn_sum_from_red"),
            (func.sum(Purchases.trn_sum_from_iss) - func.coalesce(func.sum(Purchases.trn_sum_from_red), 0)).label("diff_sum_from_iss_red")
        ).join(Products, Purchases.product_id == Products.product_id)

        subq_last_month_1 = subq_1.filter(Purchases.transaction_datetime > '2019-02-18').group_by(
            Purchases.client_id,
            Purchases.transaction_id,
            Purchases.store_id,
            "transaction_time",
            "transaction_weekday",
            Purchases.regular_points_received,
            Purchases.express_points_received,
            Purchases.regular_points_spent,
            Purchases.express_points_spent,
            Purchases.purchase_sum,
        ).subquery()

        subq_1 = subq_1.group_by(
            Purchases.client_id,
            Purchases.transaction_id,
            Purchases.store_id,
            "transaction_time",
            "transaction_weekday",
            Purchases.regular_points_received,
            Purchases.express_points_received,
            Purchases.regular_points_spent,
            Purchases.express_points_spent,
            Purchases.purchase_sum,
        ).subquery()

        subq_2 = session.query(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,

            func.sum(subq_1.c.n_alcohols).label("n_alchohol_products"),
            func.avg(subq_1.c.n_alcohols).label("avg_alchohol_products_in_purchase"),
            (func.sum(subq_1.c.n_alcohols) / func.sum(subq_1.c.n_unique_products)).label("pct_alcohol_products"),
            func.sum(subq_1.c.n_own_trademark).label("n_own_trademark_products"),
            func.avg(subq_1.c.n_own_trademark).label("avg_onw_trademark_in_purchase"),
            (func.sum(subq_1.c.n_own_trademark) / func.sum(subq_1.c.n_unique_products)).label("pct_own_trademark_products"),
            func.sum(subq_1.c.sum_netto).label("sum_sum_netto"),
            func.avg(subq_1.c.sum_netto).label("avg_sum_netto"),
            func.stddev(subq_1.c.sum_netto).label("stddev_sum_netto"),
            func.avg(subq_1.c.avg_netto).label("avg_avg_netto"),
            func.stddev(subq_1.c.sum_netto).label("stddev_avg_netto"),

            func.count(distinct(subq_1.c.transaction_id)).label('n_transactions'),
            cast(func.avg(subq_1.c.transaction_time), Time).label('avg_transaction_datetime'),
            func.avg(extract('epoch', subq_1.c.transaction_time)).label('avg_transaction_time'),
            func.stddev(extract('epoch', subq_1.c.transaction_time)).label('stddev_transaction_time'),
            mode().within_group(subq_1.c.transaction_weekday).label("mode_transaction_weekday"),
            func.sum(subq_1.c.regular_points_received).label('sum_regular_points_received'),
            func.sum(subq_1.c.express_points_received).label('sum_express_points_received'),
            func.sum(subq_1.c.regular_points_spent).label('sum_regular_points_spent'),
            func.sum(subq_1.c.express_points_spent).label('sum_express_points_spent'),
            func.avg(subq_1.c.regular_points_received).label("avg_regular_points_received"),
            func.avg(subq_1.c.express_points_received).label("avg_express_points_received"),
            func.avg(subq_1.c.regular_points_spent).label("avg_regular_points_spent"),
            func.avg(subq_1.c.express_points_spent).label("avg_express_points_spent"),
            func.stddev(subq_1.c.regular_points_received).label("stdddev_regular_points_received"),
            func.stddev(subq_1.c.express_points_received).label("stdddev_express_points_received"),
            func.stddev(subq_1.c.regular_points_spent).label("stdddev_regular_points_spent"),
            func.stddev(subq_1.c.express_points_spent).label("stdddev_express_points_spent"),
            func.sum(subq_1.c.purchase_sum).label("sum_purchase_sum"),
            func.avg(subq_1.c.purchase_sum).label("avg_purchase_sum"),
            func.stddev(subq_1.c.purchase_sum).label("stddev_purchase_sum"),
            func.count(distinct(subq_1.c.store_id)).label("n_stores"),
            func.sum(subq_1.c.n_unique_products).label("sum_n_unique_products"),
            func.sum(subq_1.c.n_products).label("sum_n_products"),
            func.sum(subq_1.c.avg_product_price).label("sum_avg_product_price"),
            func.sum(subq_1.c.sum_trn_sum_from_iss).label("sum_sum_trn_sum_from_iss"),
            func.sum(subq_1.c.sum_trn_sum_from_red).label("sum_sum_trn_sum_from_red"),
            func.sum(subq_1.c.diff_sum_from_iss_red).label("sum_diff_sum_from_iss_red"),
            func.avg(subq_1.c.n_unique_products).label("avg_n_unique_products"),
            func.avg(subq_1.c.n_products).label("avg_n_products"),
            func.avg(subq_1.c.avg_product_price).label("avg_avg_product_price"),
            func.avg(subq_1.c.sum_trn_sum_from_iss).label("avg_sum_trn_sum_from_iss"),
            func.avg(subq_1.c.sum_trn_sum_from_red).label("avg_sum_trn_sum_from_red"),
            func.avg(subq_1.c.diff_sum_from_iss_red).label("avg_diff_sum_from_iss_red"),
            func.stddev(subq_1.c.n_unique_products).label("stddev_n_unique_products"),
            func.stddev(subq_1.c.n_products).label("stddev_n_products"),
            func.stddev(subq_1.c.avg_product_price).label("stddev_avg_product_price"),
            func.stddev(subq_1.c.sum_trn_sum_from_iss).label("stddev_sum_trn_sum_from_iss"),
            func.stddev(subq_1.c.sum_trn_sum_from_red).label("stddev_sum_trn_sum_from_red"),
            func.stddev(subq_1.c.diff_sum_from_iss_red).label("stddev_diff_sum_from_iss_red"),
        ).join(subq_1, Clients.client_id == subq_1.c.client_id)

        if client_ids:
            subq_2 = subq_2.filter(Clients.client_id.in_(client_ids))

        subq_2 = subq_2.group_by(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,
        ).subquery()

        subq_last_month_2 = session.query(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,

            func.count(distinct(subq_last_month_1.c.transaction_id)).label('last_month_n_transactions'),
            cast(func.avg(subq_last_month_1.c.transaction_time), Time).label('last_month_avg_transaction_datetime'),
            func.avg(extract('epoch', subq_last_month_1.c.transaction_time)).label('last_month_avg_transaction_time'),
            func.stddev(extract('epoch', subq_last_month_1.c.transaction_time)).label('last_month_stddev_transaction_time'),
            mode().within_group(subq_last_month_1.c.transaction_weekday).label("last_month_mode_transaction_weekday"),
            func.sum(subq_last_month_1.c.regular_points_received).label('last_month_sum_regular_points_received'),
            func.sum(subq_last_month_1.c.express_points_received).label('last_month_sum_express_points_received'),
            func.sum(subq_last_month_1.c.regular_points_spent).label('last_month_sum_regular_points_spent'),
            func.sum(subq_last_month_1.c.express_points_spent).label('last_month_sum_express_points_spent'),
            func.avg(subq_last_month_1.c.regular_points_received).label("last_month_avg_regular_points_received"),
            func.avg(subq_last_month_1.c.express_points_received).label("last_month_avg_express_points_received"),
            func.avg(subq_last_month_1.c.regular_points_spent).label("last_month_avg_regular_points_spent"),
            func.avg(subq_last_month_1.c.express_points_spent).label("last_month_avg_express_points_spent"),
            func.stddev(subq_last_month_1.c.regular_points_received).label("last_month_stdddev_regular_points_received"),
            func.stddev(subq_last_month_1.c.express_points_received).label("last_month_stdddev_express_points_received"),
            func.stddev(subq_last_month_1.c.regular_points_spent).label("last_month_stdddev_regular_points_spent"),
            func.stddev(subq_last_month_1.c.express_points_spent).label("last_month_stdddev_express_points_spent"),
            func.sum(subq_last_month_1.c.purchase_sum).label("last_month_sum_purchase_sum"),
            func.avg(subq_last_month_1.c.purchase_sum).label("last_month_avg_purchase_sum"),
            func.stddev(subq_last_month_1.c.purchase_sum).label("last_month_stddev_purchase_sum"),
            func.count(distinct(subq_last_month_1.c.store_id)).label("last_month_n_stores"),
            func.sum(subq_last_month_1.c.n_unique_products).label("last_month_sum_n_unique_products"),
            func.sum(subq_last_month_1.c.n_products).label("last_month_sum_n_products"),
            func.sum(subq_last_month_1.c.avg_product_price).label("last_month_sum_avg_product_price"),
            func.sum(subq_last_month_1.c.sum_trn_sum_from_iss).label("last_month_sum_sum_trn_sum_from_iss"),
            func.sum(subq_last_month_1.c.sum_trn_sum_from_red).label("last_month_sum_sum_trn_sum_from_red"),
            func.sum(subq_last_month_1.c.diff_sum_from_iss_red).label("last_month_sum_diff_sum_from_iss_red"),
            func.avg(subq_last_month_1.c.n_unique_products).label("last_month_avg_n_unique_products"),
            func.avg(subq_last_month_1.c.n_products).label("last_month_avg_n_products"),
            func.avg(subq_last_month_1.c.avg_product_price).label("last_month_avg_avg_product_price"),
            func.avg(subq_last_month_1.c.sum_trn_sum_from_iss).label("last_month_avg_sum_trn_sum_from_iss"),
            func.avg(subq_last_month_1.c.sum_trn_sum_from_red).label("last_month_avg_sum_trn_sum_from_red"),
            func.avg(subq_last_month_1.c.diff_sum_from_iss_red).label("last_month_avg_diff_sum_from_iss_red"),
            func.stddev(subq_last_month_1.c.n_unique_products).label("last_month_stddev_n_unique_products"),
            func.stddev(subq_last_month_1.c.n_products).label("last_month_stddev_n_products"),
            func.stddev(subq_last_month_1.c.avg_product_price).label("last_month_stddev_avg_product_price"),
            func.stddev(subq_last_month_1.c.sum_trn_sum_from_iss).label("last_month_stddev_sum_trn_sum_from_iss"),
            func.stddev(subq_last_month_1.c.sum_trn_sum_from_red).label("last_month_stddev_sum_trn_sum_from_red"),
            func.stddev(subq_last_month_1.c.diff_sum_from_iss_red).label("last_month_stddev_diff_sum_from_iss_red"),
        ).outerjoin(
            subq_last_month_1, Clients.client_id == subq_last_month_1.c.client_id
        )

        if client_ids:
            subq_last_month_2 = subq_last_month_2.filter(Clients.client_id.in_(client_ids))

        subq_last_month_2 = subq_last_month_2.group_by(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,
        ).subquery()

        q = session.query(
            subq_2.c.client_id,
            subq_2.c.age,
            subq_2.c.gender,
            subq_2.c.first_issue_date,
            subq_2.c.first_redeem_date,

            subq_2.c.n_alchohol_products,
            subq_2.c.avg_alchohol_products_in_purchase,
            subq_2.c.pct_alcohol_products,
            subq_2.c.n_own_trademark_products,
            subq_2.c.pct_onw_trademark_in_purchase,
            subq_2.c.pct_own_trademark_products,
            subq_2.c.sum_sum_netto,
            subq_2.c.avg_sum_netto,
            subq_2.c.stddev_sum_netto,
            subq_2.c.avg_avg_netto,
            subq_2.c.stddev_avg_netto,

            subq_2.c.n_transactions,
            subq_2.c.avg_transaction_datetime,
            subq_2.c.avg_transaction_time,
            subq_2.c.stddev_transaction_time,
            subq_2.c.mode_transaction_weekday,
            subq_2.c.sum_regular_points_received,
            subq_2.c.sum_express_points_received,
            subq_2.c.sum_regular_points_spent,
            subq_2.c.sum_express_points_spent,
            subq_2.c.avg_regular_points_received,
            subq_2.c.avg_express_points_received,
            subq_2.c.avg_regular_points_spent,
            subq_2.c.avg_express_points_spent,
            subq_2.c.stdddev_regular_points_received,
            subq_2.c.stdddev_express_points_received,
            subq_2.c.stdddev_regular_points_spent,
            subq_2.c.stdddev_express_points_spent,
            subq_2.c.sum_purchase_sum,
            subq_2.c.avg_purchase_sum,
            subq_2.c.stddev_purchase_sum,
            subq_2.c.n_stores,
            subq_2.c.sum_n_unique_products,
            subq_2.c.sum_n_products,
            subq_2.c.sum_avg_product_price,
            subq_2.c.sum_sum_trn_sum_from_iss,
            subq_2.c.sum_sum_trn_sum_from_red,
            subq_2.c.sum_diff_sum_from_iss_red,
            subq_2.c.avg_n_unique_products,
            subq_2.c.avg_n_products,
            subq_2.c.avg_avg_product_price,
            subq_2.c.avg_sum_trn_sum_from_iss,
            subq_2.c.avg_sum_trn_sum_from_red,
            subq_2.c.avg_diff_sum_from_iss_red,
            subq_2.c.stddev_n_unique_products,
            subq_2.c.stddev_n_products,
            subq_2.c.stddev_avg_product_price,
            subq_2.c.stddev_sum_trn_sum_from_iss,
            subq_2.c.stddev_sum_trn_sum_from_red,
            subq_2.c.stddev_diff_sum_from_iss_red,

            subq_last_month_2.c.last_month_n_transactions,
            subq_last_month_2.c.last_month_avg_transaction_datetime,
            subq_last_month_2.c.last_month_avg_transaction_time,
            subq_last_month_2.c.last_month_stddev_transaction_time,
            subq_last_month_2.c.last_month_mode_transaction_weekday,
            subq_last_month_2.c.last_month_sum_regular_points_received,
            subq_last_month_2.c.last_month_sum_express_points_received,
            subq_last_month_2.c.last_month_sum_regular_points_spent,
            subq_last_month_2.c.last_month_sum_express_points_spent,
            subq_last_month_2.c.last_month_avg_regular_points_received,
            subq_last_month_2.c.last_month_avg_express_points_received,
            subq_last_month_2.c.last_month_avg_regular_points_spent,
            subq_last_month_2.c.last_month_avg_express_points_spent,
            subq_last_month_2.c.last_month_stdddev_regular_points_received,
            subq_last_month_2.c.last_month_stdddev_express_points_received,
            subq_last_month_2.c.last_month_stdddev_regular_points_spent,
            subq_last_month_2.c.last_month_stdddev_express_points_spent,
            subq_last_month_2.c.last_month_sum_purchase_sum,
            subq_last_month_2.c.last_month_avg_purchase_sum,
            subq_last_month_2.c.last_month_stddev_purchase_sum,
            subq_last_month_2.c.last_month_n_stores,
            subq_last_month_2.c.last_month_sum_n_unique_products,
            subq_last_month_2.c.last_month_sum_n_products,
            subq_last_month_2.c.last_month_sum_avg_product_price,
            subq_last_month_2.c.last_month_sum_sum_trn_sum_from_iss,
            subq_last_month_2.c.last_month_sum_sum_trn_sum_from_red,
            subq_last_month_2.c.last_month_sum_diff_sum_from_iss_red,
            subq_last_month_2.c.last_month_avg_n_unique_products,
            subq_last_month_2.c.last_month_avg_n_products,
            subq_last_month_2.c.last_month_avg_avg_product_price,
            subq_last_month_2.c.last_month_avg_sum_trn_sum_from_iss,
            subq_last_month_2.c.last_month_avg_sum_trn_sum_from_red,
            subq_last_month_2.c.last_month_avg_diff_sum_from_iss_red,
            subq_last_month_2.c.last_month_stddev_n_unique_products,
            subq_last_month_2.c.last_month_stddev_n_products,
            subq_last_month_2.c.last_month_stddev_avg_product_price,
            subq_last_month_2.c.last_month_stddev_sum_trn_sum_from_iss,
            subq_last_month_2.c.last_month_stddev_sum_trn_sum_from_red,
            subq_last_month_2.c.last_month_stddev_diff_sum_from_iss_red,
        ).join(
            subq_last_month_2, subq_2.c.client_id == subq_last_month_2.c.client_id
        )

        return [col['name'] for col in q.column_descriptions], q.all()
