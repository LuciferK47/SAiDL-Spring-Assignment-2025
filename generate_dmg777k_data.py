import json
import os
import random
from sklearn.model_selection import train_test_split

# Create the dmg777k directory if it doesn't exist
if not os.path.exists("dmg777k"):
    os.makedirs("dmg777k")

# Generate synthetic data
num_entities = 100
random.seed(42)  # For reproducibility
entities = [f"http://example.com/entity_{i}" for i in range(num_entities)]
entity_ids = list(range(num_entities))
e2i = {entity: idx for entity, idx in zip(entities, entity_ids)}
i2e = {idx: entity for entity, idx in zip(entities, entity_ids)}

# Generate triples (relationships)
triples = []
for _ in range(200):  # More triples than entities for a connected graph
    s = random.choice(entities)
    p = random.choice(["related_to", "part_of", "has_property"])
    o = random.choice(entities)
    if s != o:  # Avoid self-loops
        triples.append([s, p, o])

# Generate text and image features
texts = {idx: f"Description of {entity.split('/')[-1]} with {random.randint(5, 50)} words." 
         for idx, entity in enumerate(entities) if random.random() > 0.3}
images = {idx: f"http://example.com/images/{idx}.jpg" 
          for idx, entity in enumerate(entities) if random.random() > 0.3}

# Generate labels (binary classification for simplicity)
labels = {idx: random.randint(0, 1) for idx in entity_ids if random.random() > 0.2}

# Split data into train, valid, test (80-10-10)
all_nodes = list(entity_ids)
train_nodes, temp_nodes = train_test_split(all_nodes, test_size=0.2, random_state=42)
valid_nodes, test_nodes = train_test_split(temp_nodes, test_size=0.5, random_state=42)

# Function to create dataset for each split
def create_dataset(nodes, all_triples, all_texts, all_images, all_labels):
    dataset = {
        "e2i": {k: v for k, v in e2i.items() if v in nodes},
        "i2e": {v: k for k, v in e2i.items() if v in nodes},
        "triples": [t for t in all_triples if t[0] in [e2i[k] for k in e2i if e2i[k] in nodes] and t[2] in [e2i[k] for k in e2i if e2i[k] in nodes]],
        "texts": {k: v for k, v in all_texts.items() if k in nodes},
        "images": {k: v for k, v in all_images.items() if k in nodes},
        "labels": {k: v for k, v in all_labels.items() if k in nodes},
        "nodes": nodes  # Add nodes list for train/valid/test split
    }
    return dataset

# Create datasets
train_data = create_dataset(train_nodes, triples, texts, images, labels)
valid_data = create_dataset(valid_nodes, triples, texts, images, labels)
test_data = create_dataset(test_nodes, triples, texts, images, labels)

# Save to JSON files
for name, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
    with open(os.path.join("dmg777k", f"{name}.json"), "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {name}.json with {len(data['nodes'])} nodes")

print("JSON files created in the dmg777k directory. You can now run your original script.")