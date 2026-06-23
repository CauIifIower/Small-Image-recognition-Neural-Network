import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from create_dataset import get_all_templates
# type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


# ==================== CONFIGURATION ====================

# Define base colors
COLORS = {
    'heart': [220, 80, 120],    # Pinkish-red
    'sun': [240, 220, 100],      # Yellowish
    'snake': [80, 180, 100]     # Greenish
}

# How many variations per template
VARIATIONS_PER_TEMPLATE = {
    'heart': 200,   # 5 templates × 200 = 1000
    'sun': 170,     # 6 templates × 170 = 1020
    'snake': 130    # 8 templates × 130 = 1040
}

# ==================== COLOR FUNCTION (FIXED) ====================

def add_color(template, base_color, variation=60, rotate_image=True):
    """
    Add color to a black/white template.
    The ENTIRE shape gets ONE random color (not per-pixel).
    
    Parameters:
    - template: 10x10 array of 0s and 1s
    - base_color: [R, G, B] values (0-255)
    - variation: random variation amount (default ±60)
    
    Returns:
    - 10x10x3 colored image (0-1 range)
    """
    colored = np.zeros((10, 10, 3))
    
    # Generate ONE random color for the ENTIRE shape
    r = base_color[0] + np.random.uniform(-variation, variation)
    g = base_color[1] + np.random.uniform(-variation, variation)
    b = base_color[2] + np.random.uniform(-variation, variation)
    
    # Clip to valid range
    shape_color = [
        np.clip(r, 0, 255),
        np.clip(g, 0, 255),
        np.clip(b, 0, 255)
    ]
    
    # Rotate using NumPy (0, 90, 180, or 270 degrees)
    if rotate_image:
        k = np.random.choice([0, 1, 2, 3])  # Number of 90° rotations
        template = np.rot90(template, k=k)

        # Add random flips
        if np.random.rand() > 0.5:
            template = np.fliplr(template)

        # Add random vertical flip
        if np.random.rand() > 0.5:
            template = np.flipud(template)

    # Apply this ONE color to ALL filled pixels
    for i in range(10):
        for j in range(10):
            if template[i, j] == 1:
                colored[i, j] = shape_color
    
    return colored / 255.0

# ==================== PREVIEW FUNCTION ====================

def preview_variations(template_func, base_color, num_variations=20, variation_amount=60):
    """Preview variations of one template"""
    template = template_func()
    variations = []
    
    for i in range(num_variations):
        colored = add_color(template, base_color, variation=variation_amount)
        variations.append(colored)
    
    # Display in 4×5 grid
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < len(variations):
            ax.imshow(variations[i])
            ax.set_title(f'Var {i+1}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Preview: {num_variations} variations', fontsize=14)
    plt.tight_layout()
    plt.show()

# ==================== DATASET GENERATION ====================

def generate_full_dataset(templates, colors, variations_per_template):
    all_images = []
    all_labels = []
    
    label_map = {'heart': 0, 'sun': 1, 'snake': 2}
    
    for shape_name in ['heart', 'sun', 'snake']:
        print(f"Generating {shape_name}s...")
        
        shape_templates = templates[shape_name]
        num_variations = variations_per_template[shape_name]
        base_color = colors[shape_name]
        
        for template_func in shape_templates:
            template = template_func()
            
            for _ in range(num_variations):
                colored = add_color(template, base_color, variation=60)
                all_images.append(colored)
                all_labels.append(label_map[shape_name])
    
    images = np.array(all_images)
    labels = np.array(all_labels)
    
    print(f"\n Dataset generated")
    print(f"Total images: {len(images)}")
    print(f"Hearts: {np.sum(labels == 0)}")
    print(f"Suns: {np.sum(labels == 1)}")
    print(f"Snakes: {np.sum(labels == 2)}")
    
    return images, labels

# ==================== VISUALIZATION ====================

def show_random_samples(images, labels, samples_per_class=10):
    fig, axes = plt.subplots(3, samples_per_class, figsize=(15, 5))
    
    shape_names = ['Heart', 'Sun', 'Snake']
    
    for class_idx in range(3):
        class_images = images[labels == class_idx]
        
        for i in range(samples_per_class):
            random_idx = np.random.randint(0, len(class_images))
            axes[class_idx, i].imshow(class_images[random_idx])
            axes[class_idx, i].axis('off')
            
            if i == 0:
                axes[class_idx, i].set_ylabel(shape_names[class_idx], 
                                              fontsize=14, rotation=0, 
                                              ha='right', va='center')
    
    plt.suptitle('Random Samples from Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

# ==================== SAVING ====================

def save_dataset(images, labels, filename='dataset.pkl'):
    """Save to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump({'images': images, 'labels': labels}, f)
    print(f"\n Dataset saved to '{filename}'")

def save_example_pngs(images, labels, num_per_class=5):
    """Save example PNGs"""
    os.makedirs('dataset_examples', exist_ok=True)
    
    shape_names = ['heart', 'sun', 'snake']
    
    for class_idx in range(3):
        class_images = images[labels == class_idx]
        shape_name = shape_names[class_idx]
        
        for i in range(num_per_class):
            plt.imsave(f'dataset_examples/{shape_name}_{i+1}.png', 
                      class_images[i])
    
    print(f"Example PNGs saved to 'dataset_examples/' folder")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Get templates
    templates = get_all_templates()
    
    # STEP 1: Preview one template
    print("\n step 1: previewing variations of one template")
    print("Previewing heart template 1...")
    preview_variations(templates['heart'][0], COLORS['heart'], num_variations=20, variation_amount=60)
    
    # Wait for user approval
    proceed = input("\nDo the hearts look good? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("Adjust variation amount or template and run again.")
        exit()
    
    # STEP 2: Generate full dataset
    print("\n step 2: generating the full dataset")
    images, labels = generate_full_dataset(templates, COLORS, VARIATIONS_PER_TEMPLATE)
    
    # STEP 3: Show samples
    print("\n step 3: rewiewing random samples from the generated dataset")
    show_random_samples(images, labels, samples_per_class=10)
    
    # Wait for final approval
    approve = input("\nApprove dataset and save? (yes/no): ").strip().lower()
    
    if approve == 'yes':
        # STEP 4: Save
        print("\n=== STEP 4: SAVING ===")
        save_dataset(images, labels, 'dataset.pkl')
        save_example_pngs(images, labels, num_per_class=5)
        
        print("Dataset generated. dataset.pkl file created with images and labels.")
        print(f"Total: {len(images)} images ready for NN training")
    else:
        print("\nDataset not saved. Adjust and run again.")