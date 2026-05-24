"""
make_sample_data.py
--------------------
Generates a realistic SAMPLE e-commerce dataset that mimics the structure of the
Kaggle "Amazon Sales Dataset". We use this so the project runs out-of-the-box
even without downloading from Kaggle.

The columns match the real Kaggle dataset, so if you later drop in the real
`amazon.csv`, the rest of the project keeps working unchanged.

Run:  python src/make_sample_data.py
Output: data/amazon_sample.csv
"""

import os
import random
import numpy as np
import pandas as pd

# Make results repeatable (same "random" numbers every run)
random.seed(42)
np.random.seed(42)

N = 1500  # number of products to generate

# Realistic product categories (same style as the Amazon dataset)
CATEGORIES = [
    "Electronics|Mobiles&Accessories|Chargers",
    "Electronics|Headphones|In-Ear",
    "Computers&Accessories|Accessories|Cables",
    "Home&Kitchen|Kitchen&Dining|Cookware",
    "Electronics|HomeTheater|Speakers",
    "Computers&Accessories|Laptops",
    "Home&Kitchen|HomeAppliances|Vacuum",
    "Electronics|Cameras|Accessories",
    "OfficeProducts|OfficeElectronics|Printers",
    "Electronics|WearableTech|SmartWatches",
]

# Short product blurbs by category keyword -> used for the text/RAG part
BLURBS = {
    "Chargers": "Fast charging adapter with surge protection and a durable braided cable. Compatible with most modern devices.",
    "In-Ear": "Lightweight in-ear headphones with deep bass, noise isolation, and a built-in microphone for calls.",
    "Cables": "High-speed data and charging cable with a tangle-free nylon braid and reinforced connectors.",
    "Cookware": "Non-stick cookware made from food-grade material, even heat distribution, and an easy-to-clean surface.",
    "Speakers": "Wireless Bluetooth speaker with rich sound, long battery life, and a water-resistant design.",
    "Laptops": "Everyday laptop with a fast processor, ample storage, and a bright full-HD display for work and study.",
    "Vacuum": "Powerful vacuum cleaner with strong suction, multiple attachments, and a washable filter.",
    "Accessories": "Protective camera accessory kit with a carrying case, cleaning cloth, and lens guards.",
    "Printers": "Compact wireless printer with fast print speeds, mobile printing, and low running costs.",
    "SmartWatches": "Smartwatch with heart-rate tracking, notifications, multiple sport modes, and a week of battery life.",
}

rows = []
for i in range(N):
    category = random.choice(CATEGORIES)
    leaf = category.split("|")[-1]

    # actual_price: depends a bit on category, with noise
    base = {
        "Chargers": 800, "In-Ear": 1500, "Cables": 400, "Cookware": 2000,
        "Speakers": 3000, "Laptops": 45000, "Vacuum": 9000, "Accessories": 1200,
        "Printers": 12000, "SmartWatches": 6000,
    }[leaf]
    actual_price = max(99, int(np.random.normal(base, base * 0.4)))

    # rating between 2.5 and 5.0, skewed toward higher ratings
    rating = round(min(5.0, max(2.5, np.random.normal(4.1, 0.4))), 1)

    # rating_count: popular items get more reviews
    rating_count = int(abs(np.random.normal(5000, 8000))) + 5

    # --- The "true" relationship the model will try to learn ---
    # Discount % is driven by: lower rating -> bigger discount,
    # higher price -> slightly bigger discount, plus category effect + noise.
    cat_effect = {
        "Chargers": 5, "In-Ear": 8, "Cables": 3, "Cookware": 6,
        "Speakers": 10, "Laptops": 12, "Vacuum": 9, "Accessories": 7,
        "Printers": 15, "SmartWatches": 11,
    }[leaf]
    discount = (
        40
        - (rating - 4.0) * 18           # better rating -> less discount
        + (np.log1p(actual_price) - 7) * 2  # pricier -> a bit more discount
        + cat_effect
        + np.random.normal(0, 5)        # random noise
    )
    discount_percentage = float(min(80, max(0, round(discount, 1))))

    discounted_price = int(actual_price * (1 - discount_percentage / 100))

    blurb = BLURBS[leaf]
    name = f"{leaf} Model {1000 + i}"

    rows.append({
        "product_id": f"P{100000 + i}",
        "product_name": name,
        "category": category,
        "discounted_price": f"₹{discounted_price:,}",
        "actual_price": f"₹{actual_price:,}",
        "discount_percentage": f"{discount_percentage}%",
        "rating": rating,
        "rating_count": f"{rating_count:,}",
        "about_product": blurb,
        "review_content": f"Customers say: {blurb} Overall a {rating}-star experience.",
    })

df = pd.DataFrame(rows)

out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "amazon_sample.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows -> {out_path}")
print(df.head(3).to_string())
