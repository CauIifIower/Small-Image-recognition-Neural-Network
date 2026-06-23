from tempfile import template
import numpy as np
import matplotlib.pyplot as plt
import os


def create_heart_template_1():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))
    
    # Mark which pixels should be filled (1 = filled, 0 = empty)
    # Row 2: Fill pixels at positions 3-5 and 7-9
    template[1, 2:5] = 1
    template[1, 6:9] = 1
    
    # Row 3: Fill pixels 2 through 10
    template[2, 1:] = 1
    
    # Row 4: Fill pixels 2 through 10
    template[3, 1:] = 1
    
    # Row 5: Fill pixels 2 through 10
    template[4, 1:] = 1
    
    # Row 6: Fill pixels 3 through 9
    template[5, 2:9] = 1
    
    # Row 7: Fill pixels 4 through 8
    template[6, 3:8] = 1
    
    # Row 8: Fill pixels 5 through 7
    template[7, 4:7] = 1

    template[8, 5] = 1
    
    return template

# Test the heart template
heart1 = create_heart_template_1()
plt.imshow(heart1, cmap='gray')
plt.title('Heart Template 1')
plt.show()

def create_heart_template_2():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the heart template
heart2 = create_heart_template_2()
plt.imshow(heart2, cmap='gray')
plt.title('Heart Template 2')
plt.show()

def create_heart_template_3():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))

    template[2, 2:4] = 1
    template[2, 5:7] = 1
    template[3, 1:8] = 1
    template[4, 1:8] = 1
    template[5, 2:7] = 1
    template[6, 3:6] = 1
    template[7, 4:5] = 1
    return template

# Test the heart template
heart3 = create_heart_template_3()
plt.imshow(heart3, cmap='gray')
plt.title('Heart Template 3')
plt.show()

def create_heart_template_4():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the heart template
heart4 = create_heart_template_4()
plt.imshow(heart4, cmap='gray')
plt.title('Heart Template 4')
plt.show()

def create_heart_template_5():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the heart template
heart5 = create_heart_template_5()
plt.imshow(heart5, cmap='gray')
plt.title('Heart Template 5')
plt.show()


def create_snake_template_1():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the cloud template
snake1 = create_snake_template_1()
plt.imshow(snake1, cmap='gray')
plt.title('Snake Template 1')
plt.show()

def create_snake_template_2():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the cloud template
snake2 = create_snake_template_2()
plt.imshow(snake2, cmap='gray')
plt.title('Snake Template 2')
plt.show()

def create_snake_template_3():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))

    template[2, 1:3] = 1
    template[3, 2:7] = 1
    template[4, 5:9] = 1
    template[5, 8] = 1
    template[6, 6:8] = 1
    template[7, 1:7] = 1
    template[8, 1:3] = 1
    return template

# Test the cloud template
snake3 = create_snake_template_3()
plt.imshow(snake3, cmap='gray')
plt.title('Snake Template 3')
plt.show()  

def create_snake_template_4():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the cloud template
snake4 = create_snake_template_4()
plt.imshow(snake4, cmap='gray')
plt.title('Snake Template 4')
plt.show()

def create_snake_template_5():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the cloud template
snake5 = create_snake_template_5()
plt.imshow(snake5, cmap='gray')
plt.title('Snake Template 5')
plt.show()

def create_snake_template_6():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the cloud template
snake6 = create_snake_template_6()
plt.imshow(snake6, cmap='gray')
plt.title('Snake Template 6')
plt.show()

def create_snake_template_7():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))
    template[1, 1:6] = 1
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

# Test the cloud template
snake7 = create_snake_template_7()
plt.imshow(snake7, cmap='gray')
plt.title('Snake Template 7')
plt.show()

def create_snake_template_8():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))
    template[5, 1:9] = 1
    return template

# Test the cloud template
snake8 = create_snake_template_8()
plt.imshow(snake8, cmap='gray')  
plt.title('Snake Template 8')
plt.show()  

def create_sun_template_1():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the sun template
sun1 = create_sun_template_1()
plt.imshow(sun1, cmap='gray')
plt.title('Sun Template 1')
plt.show()

def create_sun_template_2():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the sun template
sun2 = create_sun_template_2()
plt.imshow(sun2, cmap='gray')
plt.title('Sun Template 2')
plt.show()

def create_sun_template_3():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the sun template
sun3 = create_sun_template_3()
plt.imshow(sun3, cmap='gray')
plt.title('Sun Template 3')
plt.show()

def create_sun_template_4():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the sun template
sun4 = create_sun_template_4()
plt.imshow(sun4, cmap='gray')
plt.title('Sun Template 4')
plt.show()

def create_sun_template_5():
    # Create a 10x10 grid, all zeros (black/empty)
    template = np.zeros((10, 10))

    template[2, 3:7] = 1
    template[3, 2:8] = 1
    template[4, 2:8] = 1
    template[5, 2:8] = 1
    template[6, 2:8] = 1
    template[7, 3:7] = 1

    return template

# Test the sun template
sun5 = create_sun_template_5()
plt.imshow(sun5, cmap='gray')
plt.title('Sun Template 5')
plt.show()

def create_sun_template_6():
    # Create a 10x10 grid, all zeros (black/empty)
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

# Test the sun template
sun6 = create_sun_template_6()
plt.imshow(sun6, cmap='gray')
plt.title('Sun Template 6')
plt.show()
