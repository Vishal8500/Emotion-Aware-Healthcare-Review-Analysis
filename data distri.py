import pandas as pd
import json
import matplotlib.pyplot as plt

# === Step 1: Load business data ===
business_path = r"D:\NLP\data\yelp_academic_dataset_business.json"
businesses = []
with open(business_path, encoding='utf8') as f:
    for line in f:
        businesses.append(json.loads(line))
business_df = pd.DataFrame(businesses)

# === Step 2: Define domains and keywords ===
domains = {
    "Healthcare": ["Hospital", "Doctor", "Clinic", "Medical", "Dentist", "Health", "Urgent Care"],
    "Technology": ["IT", "Software", "Electronics", "Tech", "Computer"],
    "Food & Restaurants": ["Restaurant", "Cafe", "Food", "Bar", "Pub", "Bakery"],
    "Education": ["School", "College", "University", "Education", "Training", "Academy"],
    "Retail": ["Shop", "Store", "Mall", "Boutique", "Market"],
    "Travel & Hospitality": ["Hotel", "Resort", "Tourism", "Travel", "Motel", "Inn"],
    "Automotive": ["Auto", "Car", "Garage", "Mechanic", "Tire", "Vehicle"]
}

# === Step 3: Categorize each business ===
def categorize_business(categories):
    if pd.isna(categories):
        return "Other"
    for domain, keywords in domains.items():
        if any(keyword.lower() in categories.lower() for keyword in keywords):
            return domain
    return "Other"

business_df["Domain"] = business_df["categories"].apply(categorize_business)

# === Step 4: Count businesses per domain ===
domain_counts = business_df["Domain"].value_counts().reset_index()
domain_counts.columns = ["Domain", "Count"]

# === Step 5: Plot the distribution ===
plt.figure(figsize=(10, 6))
plt.bar(domain_counts["Domain"], domain_counts["Count"])
plt.title("Business Distribution by Domain", fontsize=16)
plt.xlabel("Domain", fontsize=12)
plt.ylabel("Number of Businesses", fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Optional: Save as CSV ===
domain_counts.to_csv("domain_distribution.csv", index=False)
print("✅ Domain distribution saved as domain_distribution.csv")
