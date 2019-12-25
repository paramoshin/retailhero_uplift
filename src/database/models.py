from sqlalchemy import Column, String, Integer, DateTime, Float, Boolean, ForeignKey

from .base import Base


class Clients(Base):
    __tablename__ = "clients"

    client_id = Column("client_id", String, primary_key=True)
    first_issue_date = Column("first_issue_date", DateTime, nullable=False)
    first_redeem_date = Column("first_redeem_date", DateTime, nullable=True)
    age = Column("age", Integer, nullable=False)
    gender = Column("gender", String, nullable=False, default="U")


class Products(Base):
    __tablename__ = "products"

    product_id = Column("product_id", String, primary_key=True)
    level_1 = Column("level_1", String, nullable=False)
    level_2 = Column("level_2", String, nullable=False)
    level_3 = Column("level_3", String, nullable=False)
    level_4 = Column("level_4", String, nullable=False)
    segment_id = Column("segment_id", Integer, nullable=True)
    brand_id = Column("brand_id", String, nullable=True)
    vendor_id = Column("vendor_id", String, nullable=True)
    netto = Column("netto", Float, nullable=False)
    is_own_trademark = Column("is_own_trademark", Boolean, nullable=False)
    is_alcohol = Column("is_alcohol", Boolean, nullable=False)


class Purchases(Base):
    __tablename__ = "purchases"

    client_id = Column("client_id", String, ForeignKey("clients.client_id"), nullable=False)
    transaction_id = Column("transaction_id", String, primary_key=True, nullable=False)
    transaction_datetime = Column("transaction_datetime", DateTime, nullable=False)
    regular_points_received = Column("regular_points_received", Float, nullable=False)
    express_points_received = Column("express_points_received", Float, nullable=False)
    regular_points_spent = Column("regular_points_spent", Float, nullable=False)
    express_points_spent = Column("express_points_spent", Float, nullable=False)
    purchase_sum = Column("purchase_sum", Float, nullable=False)
    store_id = Column("store_id", String, primary_key=True, nullable=False)
    product_id = Column("product_id", String, ForeignKey("products.product_id"), primary_key=True, nullable=False)
    product_quantity = Column("product_quantity", Float, nullable=False)
    trn_sum_from_iss = Column("trn_sum_from_iss", Float, nullable=False)
    trn_sum_from_red = Column("trn_sum_from_red", Float, nullable=True)


