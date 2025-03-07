import nbformat
import base64
import weasyprint
import matplotlib.pyplot as plt
from nbconvert import HTMLExporter
from io import BytesIO
import os

# Load the notebook
notebook_path = "solution 4.ipynb"  # Change this to your actual notebook file
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Find the last code cell with output
last_code_cell = None
for cell in reversed(notebook.cells):
    if cell.cell_type == "code" and "outputs" in cell and cell["outputs"]:
        last_code_cell = cell
        break

if last_code_cell:
    html_content = "<h2>Last Code Cell Output</h2>"

    for output in last_code_cell["outputs"]:
        # Extract text output
        if "text" in output:
            html_content += f"<pre>{output['text']}</pre>"

        # Extract images (matplotlib/seaborn plots)
        if "data" in output and "image/png" in output["data"]:
            image_data = base64.b64decode(output["data"]["image/png"])
            image_path = "last_cell_graph.png"

            # Save the image
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)

            # Embed the image in the PDF content
            html_content += f'<img src="{image_path}" style="max-width:100%;">'

    # 📌 Force saving the last matplotlib figure (in case it's not captured)
    fig = plt.gcf()  # Get current figure
    if fig and fig.get_size_inches() != (0.0, 0.0):  # Check if a figure exists
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image_path = "forced_last_graph.png"
        with open(image_path, "wb") as img_file:
            img_file.write(buffer.read())

        html_content += f'<img src="{image_path}" style="max-width:100%;">'

    # Convert to PDF
    pdf_path = "last_code_cell_output.pdf"
    weasyprint.HTML(string=html_content).write_pdf(pdf_path)
    print(f"✅ Saved: {pdf_path}")

    # Clean up temporary image file
    if os.path.exists(image_path):
        os.remove(image_path)

else:
    print("⚠️ No code cell with output found in the notebook.")
