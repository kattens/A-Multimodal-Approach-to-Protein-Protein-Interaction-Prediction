import numpy as np
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\KATT\Documents\ProteinInteraPredict\newcsvforprotchains.csv")

# Clean up the 'File Name' column by removing the last 4 characters (assuming they are '.pdb')
df['File Name'] = df['File Name'].str[:-4]

# Define a function to parse the coordinate string into tuples of (x, y, z)
def parse_coordinates(coord_string):
    # Convert the string to a list of floats
    coords = [float(x) for x in coord_string.split()]
    # Ensure the total number of coordinates is a multiple of 3
    if len(coords) % 3 != 0:
        raise ValueError("The number of coordinates is not a multiple of three.")
    # Group the coordinates into tuples and return
    return [(coords[i], coords[i+1], coords[i+2]) for i in range(0, len(coords), 3)]

# Define a function to safely parse coordinates with error handling
def safe_parse_coordinates(row):
    coord_string = row['C-alpha Coordinates']
    # Check if the coordinate data is indeed a string
    if isinstance(coord_string, str):
        try:
            return parse_coordinates(coord_string)
        except ValueError:
            # Return None if an error occurs during parsing
            return None
    else:
        # Return None or handle non-string data differently
        return None

# Apply the safe parsing function to each row in the DataFrame
df['Parsed Coordinates'] = df.apply(safe_parse_coordinates, axis=1)

# Drop rows where coordinate parsing failed (i.e., 'Parsed Coordinates' is None)
df = df.dropna(subset=['Parsed Coordinates'])

# Ensure sequences have a minimum length of 20 and are not null
# Note: This line had a logical error in the original code; it was replacing 'Sequence' with a boolean rather than filtering rows.
# Here's a corrected approach:
df = df[df['Sequence'].apply(lambda x: len(x) > 20 and x is not None)]

new_df = df.copy()
new_df.to_csv(r"C:\Users\KATT\Documents\ProteinInteraPredict\Modified_df.csv", index= False)