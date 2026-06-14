import random
from datetime import datetime, timedelta
from faker import Faker
from database import get_connection, create_table

fake = Faker()

PRODUCTS = [
    "wireless headphones", "running shoes", "laptop stand", "smart watch",
    "jacket", "backpack", "phone case", "desk lamp", "keyboard", "sunglasses",
    "yoga mat", "water bottle", "notebook", "camera lens", "earbuds"
]

CATEGORIES = [
    "Refunds & Returns",
    "Delivery Delay",
    "Damaged Product",
    "Payment Issue",
    "Feedback & Praise",
]


def _delivery_delay():
    order  = random.randint(100000, 999999)
    days   = random.randint(4, 14)
    amount = round(random.uniform(30, 350), 2)
    date   = fake.date_between(start_date="-20d", end_date="-5d").strftime("%B %d")
    return random.choice([
        f"Order #{order} was supposed to arrive by {date} but it's been {days} days and I've heard nothing. Tracking page just says 'In Transit'. Paid ${amount} — this is not acceptable.",
        f"I placed order #{order} on {date} and still haven't received it after {days} days. Customer support keeps sending automated replies. I need a real update.",
        f"Where is my order #{order}? It's now {days} days past the estimated delivery date. I paid extra for faster shipping and this is the service I get?",
        f"Tracking for order #{order} hasn't updated in {days} days. Last location shown is a warehouse. Is it lost? I paid ${amount} for this.",
        f"I've been waiting {days} days for order #{order}. The carrier website just shows an error. I need to know if my ${amount} purchase is coming or not.",
        f"My delivery for order #{order} is overdue by {days} days. I need this for an upcoming event. Please escalate this and give me a real answer.",
        f"Order #{order} was marked as out for delivery {days} days ago and still nothing arrived at my door. ${amount} is a lot of money to just disappear.",
        f"It's been {days} days since my expected delivery date for order #{order}. No email, no update, nothing. I'm incredibly frustrated right now.",
        f"Placed order #{order} for ${amount} on {date}. Still not here after {days} days. Your competitor delivers in 2 days. This is really disappointing.",
        f"Order #{order} is {days} days late with zero communication from your team. I had to call the bank to check if you even charged me for a product that never arrived.",
    ])

def _damaged_product():
    order   = random.randint(100000, 999999)
    product = random.choice(PRODUCTS)
    amount  = round(random.uniform(40, 450), 2)
    return random.choice([
        f"My {product} from order #{order} arrived completely crushed. The box looked like it had been dropped from a great height. I paid ${amount} for a broken item.",
        f"Order #{order} delivered a damaged {product}. There are visible cracks and it doesn't function at all. I need a replacement shipped today.",
        f"The {product} in order #{order} was packed so poorly it arrived shattered. I have photos. This is not the quality I expected for ${amount}.",
        f"I just unboxed order #{order} and the {product} inside is clearly defective — broken seal, damaged casing. Not a refurbished item situation, this is brand new and broken.",
        f"Received order #{order} today. The {product} was wrapped in a single layer of paper. Predictably, it's dented and non-functional. Please process a return.",
        f"Order #{order}: the {product} I received has a broken {random.choice(['screen', 'clasp', 'strap', 'lid', 'handle'])}. Cannot use it. Paid ${amount} and this is what I get?",
        f"The outer box for order #{order} had a massive hole in it. The {product} inside was scratched all over. Please send a replacement or issue a refund.",
        f"My {product} (order #{order}, ${amount}) arrived with a manufacturing defect — it won't even turn on. Box was sealed so this isn't a shipping damage issue.",
        f"I'm really disappointed. The {product} in order #{order} looks nothing like the product photos and has visible damage straight out of the box. Requesting refund.",
        f"The {product} I ordered (#{order}) smells chemically burned and has clear signs of damage. I don't feel safe using it. Need an urgent replacement.",
    ])

def _payment_issue():
    order  = random.randint(100000, 999999)
    amount = round(random.uniform(40, 500), 2)
    card   = random.randint(1000, 9999)
    return random.choice([
        f"My card ending in {card} was charged ${amount} twice for order #{order}. Please reverse the duplicate transaction immediately.",
        f"I see two identical charges of ${amount} on my bank statement for order #{order}. This is a double billing error — I need the second charge reversed today.",
        f"I was not expecting a charge of ${amount} from your company. I have no order confirmation email and no order #{order} in my account history. What happened?",
        f"My checkout for order #{order} kept failing but ${amount} was still deducted from my account. I have no order but I've been charged. This needs to be fixed now.",
        f"Your payment system glitched during my order #{order}. I was shown an error page but got charged ${amount} anyway. Please refund or confirm the order.",
        f"There's an unauthorised charge of ${amount} on my account associated with order #{order}. I did not complete this purchase. Please investigate and reverse.",
        f"I applied a ${round(amount * 0.2, 2)} coupon to order #{order} but was still charged the full ${amount}. The discount was not applied at checkout.",
        f"I cancelled order #{order} within the cancellation window but the ${amount} charge is still showing on my card after {random.randint(5, 10)} business days.",
        f"Paid ${amount} via PayPal for order #{order} but the payment shows as 'pending' on PayPal and 'completed' on your site. I'm in limbo — where is my money?",
        f"I tried to split my ${amount} payment across two cards for order #{order} but both were charged the full amount. That's a ${amount * 2:.2f} overcharge total.",
    ])

def _refund_return():
    order  = random.randint(100000, 999999)
    amount = round(random.uniform(30, 300), 2)
    days   = random.randint(5, 25)
    product = random.choice(PRODUCTS)
    return random.choice([
        f"I want to return my {product} from order #{order}. It doesn't match the size chart on your website and I'd like a full refund of ${amount}.",
        f"I initiated a return request {days} days ago for order #{order} and still haven't received the prepaid label or a refund confirmation.",
        f"The {product} in order #{order} looks completely different from the website photos. I'm well within the 30-day return window — please process this.",
        f"Return for order #{order} was approved {days} days ago. Tracking shows the item was received back {days - 3} days ago. Where is my ${amount} refund?",
        f"I need to return order #{order} (${amount}). The product quality is far below what was described. How do I get my refund started?",
        f"The {product} from order #{order} broke after {random.randint(2, 10)} days of normal use. I want to return it under your quality guarantee and get a ${amount} refund.",
        f"Order #{order}: I was sent the wrong item — I ordered a {product} and received something completely different. Please send a return label and process my refund.",
        f"My return for order #{order} was submitted on {fake.date_between('-25d', '-10d').strftime('%B %d')}. Still no refund. That's ${amount} sitting unresolved for too long.",
        f"The return portal shows my #{order} return as 'received' but ${amount} hasn't been credited. It's been {days} business days. What's the hold up?",
        f"I'm trying to return order #{order} but your return form keeps throwing a server error. I'm within the return window — please help me before it expires.",
    ])

def _praise():
    order   = random.randint(100000, 999999)
    product = random.choice(PRODUCTS)
    days    = random.randint(1, 3)
    agent   = fake.first_name()
    return random.choice([
        f"Just received order #{order} — the {product} is exactly as described and arrived in only {days} days! Packaging was perfect. Will definitely order again.",
        f"I had a billing question and {agent} from your support team resolved it in under 10 minutes. Incredible response time and super professional.",
        f"The returns process for order #{order} was surprisingly smooth. Prepaid label arrived same day, refund processed in 2 business days. Thank you!",
        f"My {product} from order #{order} is absolutely outstanding quality. I was nervous ordering online but this exceeded every expectation. 5 stars.",
        f"Shoutout to your support team — {agent} helped me change my delivery address at the last minute and still got my order #{order} on time. Amazed.",
        f"Just want to say order #{order} arrived a full day early. The {product} was beautifully packaged and works perfectly. This is how e-commerce should work.",
        f"I've ordered {random.randint(3, 12)} times now and the experience keeps getting better. Order #{order} was the smoothest yet. Keep up the fantastic work.",
        f"Your live chat support is genuinely the best I've encountered. {agent} solved my issue with order #{order} in minutes without me having to repeat myself.",
        f"The {product} quality for ${round(random.uniform(40, 200), 2)} is unbeatable. Order #{order} delivered fast, zero damage, and works flawlessly. Highly recommend.",
        f"I was worried about a delayed delivery for order #{order} and reached out to support. {agent} proactively tracked it down and gave me real-time updates. Excellent.",
    ])

TEMPLATE_MAP = {
    "Delivery Delay":   _delivery_delay,
    "Damaged Product":  _damaged_product,
    "Payment Issue":    _payment_issue,
    "Refunds & Returns": _refund_return,
    "Feedback & Praise": _praise,
}

NEGATIVE_CATEGORIES = {"Delivery Delay", "Damaged Product", "Payment Issue", "Refunds & Returns"}


def _make_row(category: str) -> dict:
    """Build a single ticket dict with realistic, randomised fields."""
    is_negative = category in NEGATIVE_CATEGORIES

    if is_negative:
        # Negative tickets get low ratings with some variance
        rating = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
        status = random.choices(
            ["Open", "In Progress", "Closed"],
            weights=[40, 40, 20]
        )[0]
    else:
        # Positive / praise tickets get high ratings with some variance
        rating = random.choices([3, 4, 5], weights=[10, 35, 55])[0]
        status = "Closed"

    resolution_eta = (
        f"{random.randint(1, 24)} hours" if status != "Closed" else "Resolved"
    )

    return {
        "timestamp":         datetime.now() - timedelta(days=random.randint(0, 90)),
        "customer_name":     fake.name(),
        "category":          category,
        "issue_description": TEMPLATE_MAP[category](),  # unique each call
        "rating":            rating,
        "order_value":       round(random.uniform(10.99, 499.99), 2),
        "status":            status,
        "resolution_eta":    resolution_eta,
        "device_type":       random.choice(
            ["Mobile App", "Desktop Web", "iOS App", "Android App", "Phone Call"]
        ),
    }


def generate_and_insert_data(num_rows: int = 1500):
    """Creates the table and inserts num_rows of diverse, unique ticket data."""
    create_table()
    conn = get_connection()
    cur = conn.cursor()

    print(f"Generating {num_rows} unique tickets ...")

    for i in range(num_rows):
        row = _make_row(random.choice(CATEGORIES))
        cur.execute("""
            INSERT INTO support_tickets
                (timestamp, customer_name, category, issue_description, rating,
                 order_value, status, resolution_eta, device_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            row["timestamp"], row["customer_name"], row["category"],
            row["issue_description"], row["rating"], row["order_value"],
            row["status"], row["resolution_eta"], row["device_type"],
        ))

        if (i + 1) % 250 == 0:
            print(f"  Inserted {i + 1}/{num_rows} rows ...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done — {num_rows} diverse rows inserted into support_tickets.")


if __name__ == "__main__":
    generate_and_insert_data(1500)
