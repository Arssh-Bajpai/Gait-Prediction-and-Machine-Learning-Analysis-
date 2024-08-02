import csv
import os

def process_text_file_and_write_to_csv(filename):
    data_list = []  # List to store tuples
    if not os.path.exists(filename):
        return
    try:
        print(f"{filename} exists")

        with open(filename, 'r') as file:
            for line in file:
                # Splitting the line using '\t' as the delimiter
                row_data = line.strip().split('\t')
                # Converting the split parts into a tuple and appending it to the list
                data_list.append(tuple(row_data))
        print(f"Processed: {filename}")
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Ensure the csv_files directory exists
    output_directory = './csv_files/'
    os.makedirs(output_directory, exist_ok=True)

    # Extracting the base name without extension and creating the CSV filename
    base_name = os.path.basename(filename).split('.')[0] + '.csv'
    csv_filename = os.path.join(output_directory, base_name)

    # Writing to the CSV file
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ["Time (seconds)",
                          "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
                          "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
                          "Total_L", "Total_R"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in data_list:
                if len(item) == len(fieldnames):
                    writer.writerow(dict(zip(fieldnames, item)))
                else:
                    print(f"Warning: Data length mismatch in file {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the CSV file: {e}")

# Generate file names dynamically and process each file
for file_name_starter in ['GaCo', 'GaPt', 'JuCo', 'JuPt', 'SiCo', 'SiPt']:
    for i in range(1, 43):  # Generating numbers from 1 to 42
        for j in range(1, 43):  # Starting j from i to ensure unique combinations
            filename = f'{file_name_starter}{i:02d}_{j:02d}.txt'
            print(filename)
            full_path = os.path.join('./text_files/', filename)
            process_text_file_and_write_to_csv(full_path)