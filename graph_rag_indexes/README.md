# Instructions

### 1. Place all file types that will be uploaded to graph rag into:

- `./<index-name>/temp_input`
- Example: Within the folder `./admin/temp_input/` there is html and docx files

### 2. Run `python ./script.py <index-name>`

- This will convert all file types into txt and place them into `./<index-name>/input/` folder, then run graphrag to make the graphrag relational DB on all those new txt files
