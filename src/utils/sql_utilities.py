import sqlite3
import re

from config.config import Config
from func_timeout import func_timeout, FunctionTimedOut



def execute_query(query: str, db_id: str, timeout: int, limited_rows: int = "none") -> dict:

    """
    Executes a SQL query on the database with a timeout wrapper.

    Args:
        query (str): SQL query to be executed.
        db_id (str): Database ID for lookup.
        timeout (int): Timeout duration in seconds.

    Returns:
        dict: Execution result including connection status, number of rows, and score.
    """
        
    try:
        result = func_timeout(timeout, execute_sql, args=(query, db_id, limited_rows))
    except FunctionTimedOut:
        result = {"query": query, "connection_successful": False, "number_of_rows": 0, "data": [], "error_message": "Timeout reached", "score": 0}
    except Exception as e:
        result = {"query": query, "connection_successful": False, "number_of_rows": 0, "data": [], "error_message": str(e), "score": 0}

    return result



def execute_sql(query: str, db_id: str, limited_rows: int = "none") -> dict:

    """
    Executes a SQL query against a SQLite database.

    Args:
        query (str): SQL query to be executed.
        db_id (str): Database ID.

    Returns:
        dict: Result containing query output, status, and score.
    """
    
    conf = Config()
    config = conf.get_config()
    limited_rows = config["inference"]["limited_rows"]
    TABLES_DEV_PATH = conf.construct_path(config['dataset']['db_sqlite'])
    db_path = TABLES_DEV_PATH + f"/{db_id}" + f"/{db_id}.sqlite"
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        num_rows = len(rows)
        conn.close()
        if limited_rows == "none":
            limited_rows = rows[:]
        else:
            limited_rows = rows[:limited_rows]
        result = {"query": query, "connection_successful": True, "number_of_rows": num_rows, "data": [tuple(row) for row in limited_rows]} 
    
    except Exception as e:
        conn.close()
        result = {"query": query, "connection_successful": False, "number_of_rows": 0, "data": [], "error_message": str(e)}
    
    score = 0
    if result["connection_successful"]:
        score += 1

    if result["data"] != []:
        score += 1

    result["score"] = score / 2

    return result



def parse_response(response):
    """
    Extracts SQL code from code blocks in Markdown (```sql ... ```).

    Args:
        response (str): Full text possibly containing SQL in code block.

    Returns:
        str: The last SQL code block found, stripped of extra whitespace,
             or an empty string if no SQL block exists.
    """

    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        return ""