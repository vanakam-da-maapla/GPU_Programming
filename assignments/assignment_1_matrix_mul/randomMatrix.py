import numpy as np

# Define the matrix dimensions
rows = 1024
cols = 1024

# Generate a random matrix of integers between 0 and 100
random_matrix = np.random.randint(0, 9, size=(rows, cols))

# Save the NumPy array to a CSV file with integer formatting
# fmt='%d' ensures the numbers are written as integers, not floats
np.savetxt('public_test_cases/matrix_a.csv', random_matrix, delimiter=',', fmt='%d')

rows = 1024
cols = 1024

# Generate a random matrix of integers between 0 and 100
random_matrix = np.random.randint(0, 9, size=(rows, cols))

# Save the NumPy array to a CSV file with integer formatting
# fmt='%d' ensures the numbers are written as integers, not floats
np.savetxt('public_test_cases/matrix_b.csv', random_matrix, delimiter=',', fmt='%d')

print("Successfully generated 'random_matrix_int.csv'")