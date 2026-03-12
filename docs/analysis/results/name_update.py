import os

# --- CONFIGURATION ---
# Change this to the full path of your folder
TARGET_DIRECTORY = r'ackley_results_standard_adapter!!'

SEARCH_TERM = "Aggressive"
REPLACE_TERM = "ExtraSafe"
# ---------------------

def rename_files():
    # Verify the directory exists
    if not os.path.exists(TARGET_DIRECTORY):
        print(f"Error: The directory '{TARGET_DIRECTORY}' does not exist.")
        return

    count = 0

    # Iterate through the files in the specified directory
    for filename in os.listdir(TARGET_DIRECTORY):
        # Check if the search term is in the filename
        if SEARCH_TERM in filename:
            # Construct the new filename
            new_filename = filename.replace(SEARCH_TERM, REPLACE_TERM)

            # Create full file paths
            old_file_path = os.path.join(TARGET_DIRECTORY, filename)
            new_file_path = os.path.join(TARGET_DIRECTORY, new_filename)

            try:
                # Perform the rename
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: '{filename}' -> '{new_filename}'")
                count += 1
            except Exception as e:
                print(f"Failed to rename '{filename}': {e}")

    print(f"\nTask complete. {count} files were updated.")

if __name__ == "__main__":
    rename_files()