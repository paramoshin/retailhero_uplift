from typing import Optional, List, Tuple

from sqlalchemy import func, distinct, cast, Time, extract, within_group
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
            self, session: Session, client_ids: Optional[List[str]] = None
    ) -> Tuple[List, List]:
        subq = session.query(
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
            Purchases.product_quantity,
            Purchases.trn_sum_from_iss,
            Purchases.trn_sum_from_red,
        ).subquery()

        q = session.query(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,
            func.count(distinct(subq.c.transaction_id)).label('n_transactions'),
            cast(func.avg(subq.c.transaction_time), Time).label('avg_transaction_time'),
            func.stddev(extract('epoch', subq.c.transaction_time)).label('stddev_transaction_time'),
            mode().within_group(subq.c.transaction_weekday),
            func.sum(subq.c.regular_points_received).label('sum_regular_points_received'),
            func.sum(subq.c.express_points_received).label('sum_express_points_received'),
            func.sum(subq.c.regular_points_spent).label('sum_regular_points_spent'),
            func.sum(subq.c.express_points_spent).label('sum_express_points_spent'),
            func.avg(subq.c.regular_points_received).label("avg_regular_points_received"),
            func.avg(subq.c.express_points_received).label("avg_express_points_received"),
            func.avg(subq.c.regular_points_spent).label("avg_regular_points_spent"),
            func.avg(subq.c.express_points_spent).label("avg_express_points_spent"),
            func.stddev(subq.c.regular_points_received).label("stdddev_regular_points_received"),
            func.stddev(subq.c.express_points_received).label("stdddev_express_points_received"),
            func.stddev(subq.c.regular_points_spent).label("stdddev_regular_points_spent"),
            func.stddev(subq.c.express_points_spent).label("stdddev_express_points_spent"),
            func.sum(subq.c.purchase_sum).label("sum_purchase_sum"),
            func.avg(subq.c.purchase_sum).label("avg_purchase_sum"),
            func.stddev(subq.c.purchase_sum).label("stddev_purchase_sum"),
            func.sum(subq.c.product_quantity).label("sum_product_quantity"),
            func.avg(subq.c.product_quantity).label("avg_product_quantity"),
            func.stddev(subq.c.product_quantity).label("stddev_product_quantity"),
            func.sum(subq.c.trn_sum_from_iss).label("sum_trn_sum_from_iss"),
            func.avg(subq.c.trn_sum_from_iss).label("avg_trn_sum_from_iss"),
            func.stddev(subq.c.trn_sum_from_iss).label("stddev_trn_sum_from_iss"),
            func.sum(subq.c.trn_sum_from_red).label("sum_trn_sum_from_red"),
            func.avg(subq.c.trn_sum_from_red).label("avg_trn_sum_from_red"),
            func.stddev(subq.c.trn_sum_from_red).label("stddev_trn_sum_from_red"),
            func.count(distinct(subq.c.store_id)).label("n_stores")
        )
        if client_ids:
            q = q.filter(Clients.client_id.in_(client_ids))

        q = q.filter(Clients.client_id == subq.c.client_id).group_by(
            Clients.client_id,
            Clients.age,
            Clients.gender,
            Clients.first_issue_date,
            Clients.first_redeem_date,
        )

        return [col['name'] for col in q.column_descriptions], q.all()
