import random
from typing import List, Tuple

def random_char_insertion(text: str, p: float = 0.1) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p:
            chars.insert(i, random.choice(chars))
    return ''.join(chars)

def random_char_deletion(text: str, p: float = 0.1) -> str:
    return ''.join(char for char in text if random.random() > p)

def random_char_substitution(text: str, p: float = 0.1) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p:
            chars[i] = random.choice(chars)
    return ''.join(chars)

def random_char_swap(text: str, p: float = 0.1) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < p:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

def augment_data(data: List[Tuple[str, str]], augment_factor: int = 2) -> List[Tuple[str, str]]:
    augmented_data = []
    for noisy, clean in data:
        augmented_data.append((noisy, clean))  # Original pair
        for _ in range(augment_factor - 1):
            augmented_noisy = noisy
            augmented_noisy = random_char_insertion(augmented_noisy)
            augmented_noisy = random_char_deletion(augmented_noisy)
            augmented_noisy = random_char_substitution(augmented_noisy)
            augmented_noisy = random_char_swap(augmented_noisy)
            augmented_data.append((augmented_noisy, clean))
    return augmented_data

if __name__ == "__main__":
    from preprocess import load_data, goldset_dir

    data = load_data(goldset_dir)
    augmented_data = augment_data(data)
    print(f"Original data size: {len(data)}")
    print(f"Augmented data size: {len(augmented_data)}")
    print("Sample augmented pair:")
    print(f"Original noisy: {data[0][0]}")
    print(f"Augmented noisy: {augmented_data[-1][0]}")
    print(f"Clean: {augmented_data[-1][1]}")