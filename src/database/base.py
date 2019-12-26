import datetime
from contextlib import contextmanager
from functools import wraps

from sqlalchemy import String, DateTime, Integer, event
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()


def validate_int(value):
    if value is None:
        return value
    if isinstance(value, str):
        value = int(value)
    if not isinstance(value, int):
        raise ValueError("Value is not int")
    return value


def validate_string(value):
    if value is None:
        return value
    if not isinstance(value, int) and not isinstance(value, str):
        raise ValueError(f"Value is not 'int' or 'str': {type(value)}")
    return str(value)


def validate_datetime(value):
    if value is None:
        return value
    if not isinstance(value, datetime.datetime) and not isinstance(value, str):
        raise ValueError("Value is not 'datetime.datetime' or 'str'")
    return value


validators = {Integer: validate_int, String: validate_string, DateTime: validate_datetime}


# this event is called whenever an attribute
# on a class is instrumented
@event.listens_for(Base, "attribute_instrument")
def configure_listener(class_, key, inst):
    if not hasattr(inst.property, "columns"):
        return

    # this event is called whenever a "set"
    # occurs on that instrumented attribute
    @event.listens_for(inst, "set", retval=True)
    def set_(instance, value, oldvalue, initiator):
        validator = validators.get(inst.property.columns[0].type.__class__)
        if validator:
            return validator(value)
        else:
            return value


class SessionFactory(object):
    def __init__(self, dialect, user, password, host, port, database_name):
        self.engine = self.init_engine(dialect, user, password, host, port, database_name)
        self._SessionFactory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def init_engine(dialect, user, password, host, port, database_name):
        engine = create_engine(
            "{}://{}:{}@{}:{}/{}".format(dialect, user, password, host, port, database_name)
        )
        return engine

    @staticmethod
    def init_database(engine, tables=None):
        if not tables:
            Base.metadata.create_all(engine)
        else:
            Base.metadata.create_all(engine, tables=tables)

    def get_factory(self):
        return self._SessionFactory

    def get_session(self):
        return self._SessionFactory()


@contextmanager
def session_scope(Session):
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as exception:
        session.rollback()
        session.close()
        raise exception


def with_session(f):
    """ add session keyword argument"""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if isinstance(kwargs.get("session"), Session):
            return f(self, *args, **kwargs)
        with session_scope(self.session_factory.get_factory()) as session:
            kwargs.update({"session": session})
            return f(self, *args, **kwargs)

    return wrapper
