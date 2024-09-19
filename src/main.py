from preprocess import load_data, analyze_data, preprocess_data, goldset_dir
from augment import augment_data

if __name__ == "__main__":
    raw_data = load_data(goldset_dir)
    analyze_data(raw_data)
    
    augmented_data = augment_data(raw_data)
    print(f"Original data size: {len(raw_data)}")
    print(f"Augmented data size: {len(augmented_data)}")
    
    processed_data = preprocess_data(augmented_data)
    print(f"Processed {len(processed_data)} pairs")
    print("Sample processed pair:")
    print(f"Noisy: {processed_data[0][0][:50]}")
    print(f"Clean: {processed_data[0][1][:50]}")