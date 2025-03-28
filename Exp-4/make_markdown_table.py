import pandas as pd
import os

# Define the folder where results are stored
folder = "Results"

# Load results from CSV
csv_file = f"{folder}/training_results.csv"
df = pd.read_csv(csv_file) 

# Start writing the Markdown table
markdown_content = "| Hidden Layers | Learning Rate | Final Loss | Final Accuracy | Test Accuracy | Execution Time | Results Plot |\n"
markdown_content += "|--------------|--------------|------------|----------------|--------------|---------------|--------------|\n"

# Loop through DataFrame and format each row as a Markdown table row
for _, row in df.iterrows():
    hidden_layers = row["Hidden Layers"]
    learning_rate = row["Learning Rate"]
    final_loss = row["Final Loss"]
    final_accuracy = row["Final Accuracy"]
    test_accuracy = row["Test Accuracy"]
    execution_time = row["Execution Time"]

    # # Construct the corresponding plot file path
    # subfolder = f"{folder}/relu_{hidden_layers}_{learning_rate}"
    # plot_file = f"{subfolder}/results_{hidden_layers}_{learning_rate}.png"

    # Markdown image syntax for embedding
    # plot_md = f"![img]({plot_file})"

    # Append row to markdown table
    markdown_content += f"| {hidden_layers} | {learning_rate} | {final_loss:.4f} | {final_accuracy:.4f} | {test_accuracy:.4f} | {execution_time:.2f} sec |\n"

# Save the Markdown content to a file
markdown_file = f"training_results.md"
with open(markdown_file, "w") as md_file:
    md_file.write(markdown_content)

print(f"Markdown table saved to {markdown_file}")
