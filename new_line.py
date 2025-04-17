import time
import json

# Path to the file you're monitoring
file_path = "data.txt"  # replace with your actual file path

def monitor_file(file_path):
    # Open the file initially to read the first line (number of objects)
    with open(file_path, "r") as f:
        num_objects = int(f.readline().strip())  # Read number of expected object lines
        print(f"Number of objects: {num_objects}")

        # Read any lines already present (after the first line)
        lines = f.readlines()
        current_line_count = len(lines)  # Track how many object lines have already been added

    # Keep checking the file until we have all the expected object lines
    while current_line_count < num_objects:
        # Re-open the file to check for new content
        with open(file_path, "r") as f:
            f.readline()  # Skip the first line (it's just the object count)
            all_lines = f.readlines()  # Read all object lines

            # Get only the new lines that were added since last check
            new_lines = all_lines[current_line_count:]

            # If new lines were found
            if new_lines:
                for new_line in new_lines:
                    new_line = new_line.strip()  # Clean up newline characters

                    new_line = json.loads(new_line)
                    print(type(new_line))
                    # Print the new line that was added (coordinates or object data)
                    print(f"New line added: {new_line}")

                    # Optionally store it in a variable if you need to use it
                    latest_coordinates = new_line
                    # You can now do something with latest_coordinates
                    # return latest_coordinates

                # Update how many lines we've seen so far
                current_line_count += len(new_lines)
                return latest_coordinates

        # Wait a bit before checking again (1 second delay)
        time.sleep(1)

# Call the function to start monitoring the file
# print('latest coordinate: ',monitor_file(file_path))
