from pathlib import Path

from datasets import load_dataset

output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(exist_ok=True)

# Step 1: Stream the InfoBayAI English STEM Textbook dataset
print("Streaming InfoBayAI English STEM Dataset...")
dataset = load_dataset("InfoBayAI/English-STEM-Textbook-Dataset", split="train", streaming=True)

# Step 2: Establish rigid topic tracking boundaries
math_keywords = {"calculus", "matrix", "derivative", "integral", "theorem", "equation", "algebra", "probability"}
physics_keywords = {"mechanics", "quantum", "velocity", "thermodynamics", "momentum", "relativity", "gravity", "kinematics"}

math_count = 0
physics_count = 0

# Open isolated files to store the raw textbook data chunks
with open(output_dir / "infobay_math.txt", "w", encoding="utf-8") as f_math, \
     open(output_dir / "infobay_physics.txt", "w", encoding="utf-8") as f_phys:
    for row in dataset:
        text = row["text"].lower()

        # Isolate and save Math Textbook prose
        if any(kw in text for kw in math_keywords):
            f_math.write(row["text"] + "\n\n")
            math_count += 1

        # Isolate and save Physics Textbook prose
        elif any(kw in text for kw in physics_keywords):
            f_phys.write(row["text"] + "\n\n")
            physics_count += 1

print(f"Data separation successful! Saved {math_count} Math entries and {physics_count} Physics entries.")
