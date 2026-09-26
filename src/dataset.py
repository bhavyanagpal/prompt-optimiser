"""
Text-to-SQL examples over the schema defined in db.py.

Split discipline matters here: TRAIN_SET is the only data the optimizer's
search loop is allowed to see. TEST_SET is held out and touched exactly
once per run, to report the final baseline-vs-optimized comparison. This
mirrors real ML practice and is worth calling out explicitly in interviews
-- it's the difference between "I tuned a prompt" and "I ran an experiment".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    id: int
    question: str
    gold_sql: str


TRAIN_SET = [
    Example(1, "List the names of all customers from India.",
            "SELECT name FROM customers WHERE country = 'India';"),
    Example(2, "What is the total price of all products in the Electronics category?",
            "SELECT SUM(price) FROM products WHERE category = 'Electronics';"),
    Example(3, "How many orders have status 'completed'?",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed';"),
    Example(4, "List customer names along with the number of orders they have placed.",
            "SELECT c.name, COUNT(o.order_id) FROM customers c "
            "LEFT JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id, c.name;"),
    Example(5, "What is the most expensive product in the Furniture category?",
            "SELECT name FROM products WHERE category = 'Furniture' "
            "ORDER BY price DESC LIMIT 1;"),
    Example(6, "List all products that have never appeared in any order.",
            "SELECT p.name FROM products p WHERE p.product_id NOT IN "
            "(SELECT product_id FROM order_items);"),
    Example(7, "What is the average price of products, grouped by category?",
            "SELECT category, AVG(price) FROM products GROUP BY category;"),
    Example(8, "Which customers placed more than one completed order?",
            "SELECT c.name FROM customers c JOIN orders o "
            "ON c.customer_id = o.customer_id WHERE o.status = 'completed' "
            "GROUP BY c.customer_id, c.name HAVING COUNT(*) > 1;"),
    Example(9, "List order ids and the total quantity of items in each order.",
            "SELECT order_id, SUM(quantity) FROM order_items "
            "GROUP BY order_id;"),
    Example(10, "What is the total revenue from completed orders only? "
                "(quantity * product price, only orders with status completed)",
            "SELECT SUM(oi.quantity * p.price) FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.status = 'completed';"),
]

TEST_SET = [
    Example(101, "List the names of customers from the USA.",
            "SELECT name FROM customers WHERE country = 'USA';"),
    Example(102, "How many products are in the Stationery category?",
            "SELECT COUNT(*) FROM products WHERE category = 'Stationery';"),
    Example(103, "What is the cheapest product overall?",
            "SELECT name FROM products ORDER BY price ASC LIMIT 1;"),
    Example(104, "List each customer's name with their total number of completed orders "
                 "(customers with zero completed orders should not appear).",
            "SELECT c.name, COUNT(o.order_id) FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "WHERE o.status = 'completed' GROUP BY c.customer_id, c.name;"),
    Example(105, "Which order had the highest total quantity of items?",
            "SELECT order_id FROM order_items GROUP BY order_id "
            "ORDER BY SUM(quantity) DESC LIMIT 1;"),
    Example(106, "What is the total revenue (quantity * price) from refunded orders?",
            "SELECT SUM(oi.quantity * p.price) FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.status = 'refunded';"),
]
