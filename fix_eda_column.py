import json

notebook_path = 'EDA_Preprocessing.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the wrong column name
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Join source lines to search across them easily, or check line by line
        source_text = "".join(source)
        if "df['cleaned_text'] = df['label'].apply(clean_text)" in source_text:
            print("Found target cell. Fixing column name...")
            new_source = []
            for line in source:
                if "df['cleaned_text'] = df['label'].apply(clean_text)" in line:
                    new_source.append(line.replace("df['label']", "df['Post Content']"))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break
        # Also check if it was already correct or slightly different, but the user request implies it's currently wrong or they want to insert this.
        # Actually, looking at previous file views, the code was:
        # df['cleaned_text'] = df['Post Content'].apply(clean_text)
        # The user might have pasted a snippet they *want* to use but with a wrong column, OR they modified the file and broke it.
        # I will assume I need to ensure it uses 'Post Content'.

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully.")
