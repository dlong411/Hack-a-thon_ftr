import os
from dotenv import load_dotenv
load_dotenv()

from db import get_conn, init_db


def test_db_connection():
    conn = get_conn()
    assert conn is not None
    conn.close()


def test_init_db_runs():
    # Should not raise
    init_db()
