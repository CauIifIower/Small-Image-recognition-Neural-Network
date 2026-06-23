import numpy as np
import matplotlib.pyplot as plt

# ==================== HEART TEMPLATES ====================

def create_heart_template_1():
    template = np.zeros((10, 10))
    template[1, 2:5] = 1
    template[1, 6:9] = 1
    template[2, 1:] = 1
    template[3, 1:] = 1
    template[4, 1:] = 1
    template[5, 2:9] = 1
    template[6, 3:8] = 1
    template[7, 4:7] = 1
    template[8, 5] = 1
    return template

def create_heart_template_2():
    template = np.zeros((10, 10))
    template[3, 2:4] = 1
    template[3, 5:7] = 1
    template[4, 1] = 1
    template[4, 4] = 1
    template[4, 7] = 1
    template[5, 1] = 1
    template[5, 7] = 1
    template[6, 2] = 1
    template[6, 6] = 1
    template[7, 3] = 1
    template[7, 5] = 1
    template[8, 4] = 1
    return template

def create_heart_template_3():
    template = np.zeros((10, 10))
    template[2, 2:4] = 1
    template[2, 5:7] = 1
    template[3, 1:8] = 1
    template[4, 1:8] = 1
    template[5, 2:7] = 1
    template[6, 3:6] = 1
    template[7, 4:5] = 1
    return template

def create_heart_template_4():
    template = np.zeros((10, 10))
    template[1, 1:4] = 1
    template[1, 5:8] = 1
    template[2, 0] = 1
    template[2, 4] = 1
    template[2, 8] = 1
    template[3, 0] = 1
    template[3, 8] = 1
    template[4, 0] = 1
    template[4, 8] = 1
    template[5, 1] = 1
    template[5, 7] = 1
    template[6, 2] = 1
    template[6, 6] = 1
    template[7, 3] = 1
    template[7, 5] = 1
    template[8, 4] = 1
    return template

def create_heart_template_5():
    template = np.zeros((10, 10))
    template[2, 2:5] = 1
    template[2, 7:9] = 1
    template[3, 1:3] = 1
    template[3, 4:] = 1
    template[4, 1:3] = 1
    template[4, 5:7] = 1
    template[4, 9] = 1
    template[5, 2] = 1
    template[5, 8:] = 1
    template[6, 2:4] = 1
    template[6, 7:9] = 1
    template[7, 3:8] = 1
    template[8, 5:7] = 1
    return template

# ==================== SNAKE TEMPLATES ====================

def create_snake_template_1():
    template = np.zeros((10, 10))
    template[1, 1:9] = 1
    template[2, 1] = 1
    template[2, 7:9] = 1
    template[3, 1:6] = 1
    template[4, 5] = 1
    template[5, 1:6] = 1
    template[6, 1] = 1
    template[7, 1:3] = 1
    template[8, 2:4] = 1
    template[9, 3:5] = 1
    return template

def create_snake_template_2():
    template = np.zeros((10, 10))
    template[1, 1:9] = 1
    template[2:5, 1] = 1
    template[2:5, 8] = 1
    template[5, 1:6] = 1
    template[5, 8] = 1
    template[6, 5] = 1
    template[6, 8] = 1
    template[7, 1:6] = 1
    template[7, 8] = 1
    template[8, 1:3] = 1
    template[8, 8] = 1
    return template

def create_snake_template_3():
    template = np.zeros((10, 10))
    template[2, 1:3] = 1
    template[3, 2:7] = 1
    template[4, 5:9] = 1
    template[5, 8] = 1
    template[6, 6:8] = 1
    template[7, 1:7] = 1
    template[8, 1:3] = 1
    return template

def create_snake_template_4():
    template = np.zeros((10, 10))
    template[0, 1] = 1
    template[1, 1:9] = 1
    template[2, 8] = 1
    template[3, 1] = 1
    template[3, 3:9] = 1
    template[4, 1] = 1
    template[4, 3] = 1
    template[5, 1] = 1
    template[5, 3:9] = 1
    template[6, 1] = 1
    template[6, 8] = 1
    template[7, 1:9] = 1
    return template

def create_snake_template_5():
    template = np.zeros((10, 10))
    template[0, 4:7] = 1
    template[1, 3:5] = 1
    template[1, 6:9] = 1
    template[2, 3:5] = 1
    template[2, 8] = 1
    template[3, 3:5] = 1
    template[4, 4:6] = 1
    template[5, 5:7] = 1
    template[6, 0:2] = 1
    template[6, 6] = 1
    template[7, 1:7] = 1
    return template

def create_snake_template_6():
    template = np.zeros((10, 10))
    template[1, 1:9] = 1
    template[2, 1] = 1
    template[2, 4:7] = 1
    template[2, 8] = 1
    template[3, 1] = 1
    template[3, 3:5] = 1
    template[3, 6] = 1
    template[3, 8] = 1
    template[4, 1] = 1
    template[4, 3:7] = 1
    template[4, 8] = 1
    template[5, 1:3] = 1
    template[5, 4:9] = 1
    template[6, 2] = 1
    template[6, 4] = 1
    template[7, 2] = 1
    template[7, 4] = 1
    template[7, 8] = 1
    template[8, 1] = 1
    template[8, 4:9] = 1
    return template

def create_snake_template_7():
    template = np.zeros((10, 10))
    template[1, 1:6] = 1
    template[2, 1] = 1
    template[2, 5:7] = 1
    template[3, 6:8] = 1
    template[4, 3:7] = 1
    template[5, 2:4] = 1
    template[5, 8] = 1
    template[6, 2] = 1
    template[6, 5:9] = 1
    template[7, 2:7] = 1
    return template

def create_snake_template_8():
    template = np.zeros((10, 10))
    template[5, 1:9] = 1
    return template

# ==================== SUN TEMPLATES ====================

def create_sun_template_1():
    template = np.zeros((10, 10))
    template[0, 4] = 1
    template[0, 7] = 1
    template[1, 0] = 1
    template[1, 4] = 1
    template[1, 6] = 1
    template[2, 1] = 1
    template[2, 8] = 1
    template[3, 3:7] = 1
    template[4, 0] = 1
    template[4, 2:7] = 1
    template[4, 8] = 1
    template[4, 9] = 1
    template[5, 2:7] = 1
    template[6, 0:2] = 1
    template[6, 3:7] = 1
    template[6, 8] = 1
    template[7, 9] = 1
    template[8, 1] = 1
    template[8, 3] = 1
    template[8, 5] = 1
    template[8, 7] = 1
    template[9, 0] = 1
    template[9, 3] = 1
    template[9, 5] = 1
    template[9, 8] = 1
    return template

def create_sun_template_2():
    template = np.zeros((10, 10))
    template[0, 4] = 1
    template[1, 4] = 1
    template[1, 7] = 1
    template[2, 1] = 1
    template[2, 6] = 1
    template[3, 2] = 1
    template[3, 4:6] = 1
    template[4, 3:7] = 1
    template[4, 8:10] = 1
    template[5, 0:2] = 1
    template[5, 3:7] = 1
    template[6, 4:6] = 1
    template[6, 7] = 1
    template[7, 3] = 1
    template[7, 8] = 1
    template[8, 2] = 1
    template[8, 5] = 1
    template[9, 5] = 1
    return template

def create_sun_template_3():
    template = np.zeros((10, 10))
    template[0, 3:5] = 1
    template[1, 0:2] = 1
    template[1, 4] = 1
    template[1, 7] = 1
    template[2, 0:3] = 1
    template[2, 6:8] = 1
    template[3, 1:3] = 1
    template[3, 4:6] = 1
    template[4, 3:10] = 1
    template[5, 3:7] = 1
    template[6, 0:3] = 1
    template[6, 4:7] = 1
    template[7, 0] = 1
    template[7, 7:10] = 1
    template[8, 3:5] = 1
    template[8, 9] = 1
    template[9, 3:5] = 1
    return template

def create_sun_template_4():
    template = np.zeros((10, 10))
    template[0, 4] = 1
    template[1, 1] = 1
    template[1, 4] = 1
    template[1, 8] = 1
    template[2, 2] = 1
    template[2, 4] = 1
    template[2, 7] = 1
    template[3, 3] = 1
    template[3, 6] = 1
    template[4, 4:6] = 1
    template[4, 7:10] = 1
    template[5, 0:3] = 1
    template[5, 4:6] = 1
    template[6, 3] = 1
    template[6, 6] = 1
    template[7, 2] = 1
    template[7, 5] = 1
    template[7, 7] = 1
    template[8, 1] = 1
    template[8, 5] = 1
    template[8, 8] = 1
    template[9, 5] = 1
    return template

def create_sun_template_5():
    template = np.zeros((10, 10))
    template[2, 3:7] = 1
    template[3, 2:8] = 1
    template[4, 2:8] = 1
    template[5, 2:8] = 1
    template[6, 2:8] = 1
    template[7, 3:7] = 1
    return template

def create_sun_template_6():
    template = np.zeros((10, 10))
    template[1, 3] = 1
    template[1, 5] = 1
    template[1, 7] = 1
    template[2, 2] = 1
    template[3, 4:7] = 1
    template[3, 8] = 1
    template[4, 1] = 1
    template[4, 3:8] = 1
    template[5, 3:8] = 1
    template[5, 9] = 1
    template[6, 1] = 1
    template[6, 3:7] = 1
    template[7, 8] = 1
    template[8, 2] = 1
    template[8, 4] = 1
    template[8, 6] = 1
    return template

# ==================== COLOR FUNCTION ====================

def add_color(template, base_color, variation=30):
    """
    Add color to a black/white template with variations.
    
    Parameters:
    - template: 10x10 array of 0s and 1s
    - base_color: [R, G, B] values (0-255)
    - variation: random variation amount (default ±30)
    
    Returns:
    - 10x10x3 colored image (0-1 range)
    """
    colored = np.zeros((10, 10, 3))
    
    for i in range(10):
        for j in range(10):
            if template[i, j] == 1:
                r = base_color[0] + np.random.uniform(-variation, variation)
                g = base_color[1] + np.random.uniform(-variation, variation)
                b = base_color[2] + np.random.uniform(-variation, variation)
                
                colored[i, j] = [
                    np.clip(r, 0, 255),
                    np.clip(g, 0, 255),
                    np.clip(b, 0, 255)
                ]
    
    return colored / 255.0

# ==================== TEMPLATE ORGANIZER ====================

def get_all_templates():
    """Returns dictionary of all template functions"""
    templates = {
        'heart': [
            create_heart_template_1, create_heart_template_2,
            create_heart_template_3, create_heart_template_4,
            create_heart_template_5
        ],
        'sun': [
            create_sun_template_1, create_sun_template_2,
            create_sun_template_3, create_sun_template_4,
            create_sun_template_5, create_sun_template_6
        ],
        'snake': [
            create_snake_template_1, create_snake_template_2,
            create_snake_template_3, create_snake_template_4,
            create_snake_template_5, create_snake_template_6,
            create_snake_template_7, create_snake_template_8
        ]
    }
    return templates