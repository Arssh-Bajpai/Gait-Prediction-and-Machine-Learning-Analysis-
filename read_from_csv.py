def process_text_file(filename):
    """
    Processes a text file with two columns per row, converting each row into a tuple,
    and stores all tuples in a list.

    Parameters:
    filename (str): The path to the text file.

    Returns:
    list: A list of tuples, where each tuple represents a row from the file.
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
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return data_list

# Example usage
filename = './gait-in-aging-and-disease-database-1.0.0/o1-76-si.txt'   # Replace 'example.txt' with the path to your text file
result = process_text_file(filename)
if result is not None:
    print(result)

