import csv


def process_text_file_and_write_to_csv(filename):
    """
    Processes a text file with two columns per row, converting each row into a tuple,
    and writes the data to a new CSV file with the same base name but a .csv extension.
    The CSV file will have columns "time (seconds)" and "stride interval (seconds)".

    Parameters:
    filename (str): The path to the text file.

    Returns:
    None
    """
    data_list = []  # List to store tuples

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Splitting the line using '\t' as the delimiter
                row_data = line.strip().split('\t')
                # Converting the split parts into a tuple and appending it to the list
                data_list.append(tuple(row_data))
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Extracting the base name without extension and creating the CSV filename
    base_name = filename.split('/')[-1].split('.')[0] + '.csv'
    csv_filename = f"./{base_name}"

    # Writing to the CSV file
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ["time (seconds)", "stride interval (seconds)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in data_list:
                writer.writerow({"time (seconds)": item[0], "stride interval (seconds)": item[1]})
    except IOError as e:
        print(f"An error occurred while writing to the CSV file: {e}")

# Example usage
filenames = [
    'o1-76-si.txt',
    'o2-74-si.txt',
    'o3-75-si.txt',
    'o4-77-si.txt',
    'o5-71-si.txt',
    'pd1-si.txt',
    'pd2-si.txt',
    'pd3-si.txt',
    'pd4-si.txt',
    'pd5-si.txt',
    'y1-23-si.txt',
    'y2-29-si.txt',
    'y3-23-si.txt',
    'y4-21-si.txt',
    'y5-26-si.txt'
]

for filename in filenames:
    full_path = f'./gait-in-aging-and-disease-database-1.0.0/{filename}'
    process_text_file_and_write_to_csv(full_path)

