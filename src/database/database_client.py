from typing import Optional

from .base import SessionFactory, Base, with_session


class DatabaseClient(object):
    def __init__(self, user: str, password: str, host: str, port: int, database_name: str):
        dialect = "postgresql"
        self.session_factory = SessionFactory(dialect, user, password, host, port, database_name)

    @with_session
    def exists(self, model: Base, session=None, **kwargs) -> bool:
        query = session.query(model)
        if kwargs:
            query = query.filter_by(**kwargs)
        return session.query(query.exists()).scalar()

    @staticmethod
    def _query_get(model: Base, session, order_by: Optional[str], desc: bool = True, **kwargs):
        query = session.query(model)
        if kwargs:
            query = query.filter_by(**kwargs)
        if order_by:
            order = getattr(model, order_by)
            if desc:
                order = order.desc()
            query = query.order_by(order)
        return query

    @with_session
    def get_first(
        self, model: Base, order_by: Optional[str] = None, desc: bool = True, session=None, **kwargs
    ):
        query = self._query_get(
            model=model, order_by=order_by, desc=desc, session=session, **kwargs
        )
        return query.first()

    @with_session
    def get_all(
        self, model: Base, order_by: Optional[str] = None, desc: bool = True, session=None, **kwargs
    ):
        query = self._query_get(
            model=model, order_by=order_by, desc=desc, session=session, **kwargs
        )
        return query.all()

    @with_session
    def insert(self, model: Base, obj, check_exist: bool = False, session=None, **kwargs) -> Base:
        if not isinstance(obj, model):
            raise TypeError(f"Object is not {model}: {type(obj)}")

        if not check_exist or not self.exists(model=model, session=session, **kwargs):
            session.add(obj)
            session.commit()
        return obj

    @with_session
    def update(self, model: Base, new_data: dict, session=None, **kwargs):
        session.query(model).filter_by(**kwargs).update(new_data)
