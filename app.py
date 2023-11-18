from flask import Flask, request, jsonify, render_template
import os
import json
from datetime import datetime
from sentence_similarity import BERT_similarity,GPT_similarity,TF_similarity
from evaluation import Eva

app = Flask(__name__)


#These three global variables are for POST data and form for same route. 
selected_function=None
arg=None
gen=None

# 1 Home
@app.route('/')
def home():
    return render_template('index.html')

# 2 The main merging process
@app.route('/GPTthemes', methods=['POST'])
def process_data():
    global selected_function,arg,gen 
    if request.is_json:
        data = request.json
        titles_list = [title.replace('/', '') for title in data['titles'].values()]
    else:
        selected_function = request.form.get('function-select')
        arg = request.form.get('args-input')
        gen = request.form.get('gen-input')
        if ("humanfeed" in arg):
            selected_function='GPT'
        print(selected_function,arg)
        return jsonify({"message": "First part of data received"})
    grouped_titles = []
    if selected_function == 'BERT': 
        if not arg.strip():
            arg=0.75
        grouped_titles = BERT_similarity(titles_list, arg)
    elif selected_function == 'GPT':
        if not arg.strip():
            arg="two"
        grouped_titles = GPT_similarity(titles_list, arg)
    elif selected_function == 'TF':
        if not arg.strip():
            arg=2
        grouped_titles = TF_similarity(titles_list, arg) 
    #print(grouped_titles)
    data['theme_attributes']=merging_titles(data,grouped_titles)

    ## Evaluation
    # Extract all values associated with the key
    titles_values = list(data["titles"].values()) 
    path=os.getcwd()+ '/golden_data/golden_eva.json'  
    if os.path.exists(path):
        Eva(titles_values,grouped_titles)
    return jsonify(data)

def merging_titles(data,grouped_titles):    
    title_mapping = {}
    for group in grouped_titles:
        if len(group) > 1:
            # List for storing the matched pairs
            matched_pairs = []
            # Iterate through the dictionary and compare with the given list
            for key, value in data['titles'].items():
                if value in group:
                    matched_pairs.append([key, value])
            # Output matched pairs
            title_mapping[group[0]] = {"merged": matched_pairs}
    new_theme_attributes = {}
    for key, title in data['titles'].items():
        # If the title is the first item of any group, use the corresponding 'merged' mapping
        if title in title_mapping:
            new_theme_attributes[key] = title_mapping[title]
        else:
            new_theme_attributes[key] = {}
    return new_theme_attributes


# 3 GPT generating new data
@app.route('/generateData', methods=['POST'])
def generateData():
    global selected_function,arg,gen 
    if request.is_json:
        data = request.json
    else:
        selected_function = request.form.get('item0')
        arg = request.form.get('item1')
        gen = request.form.get('item2')
        if gen != None:
            gen=gen+" topic"
        return jsonify({"message": "First part of data received"})
    titles_list = GPT_similarity(data, gen) 
    grouped_titles = []
    if selected_function == 'BERT': 
        if not arg.strip():
            arg=0.75
        grouped_titles = BERT_similarity(titles_list, arg)
    elif selected_function == 'GPT':
        if not arg.strip():
            arg="two"
        grouped_titles = GPT_similarity(titles_list, arg)
    elif selected_function == 'TF':
        if not arg.strip():
            arg=2
        grouped_titles = TF_similarity(titles_list, arg)  

    def create_key(phrase):
        return phrase.replace(" ", "_").replace("-", "_").lower()

    # Create the JSON structure
    data = {
        "theme_attributes": {},
        "titles": {}
    }

    for phrase in titles_list:
        key = create_key(phrase)
        data["theme_attributes"][key] = {} 
        data["titles"][key] = phrase 
    data['theme_attributes']=merging_titles(data,grouped_titles)
    # Evaluation
    # Extract all values associated with the key
    titles_values = list(data["titles"].values()) 
    path=os.getcwd()+ '/golden_data/golden_eva.json'  
    if os.path.exists(path):
        Eva(titles_values,grouped_titles) 
    return jsonify(data)

# 4 Save Human Feedback data
@app.route('/save-hf-json', methods=['POST'])
def save_hf_json():
    data = request.json
    print(data,"save hf json")
    # Current timestamp
    current_time = datetime.now()

    # Formatting the date and time in the desired format (day/month/year-hour:minute:second)
    formatted_time = str(current_time.strftime("%d-%m-%Y-%H:%M:%S"))
    # Existing dictionary (if any)
    data_to_save = {}

    # Format the current time
    formatted_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    path=os.getcwd()+ '/golden_data'
    file_path = path+ '/golden_hf.json' 
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data_to_save = json.load(file)
    else:
        data_to_save = {}
        
    # Add new entry with the formatted time as the key
    data_to_save[formatted_time] = data
    
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file,indent=4)
    
    # Add new title to the grouped list if there is new similar title.
    def merge_lists_with_common_elements(list_of_lists):
        merged = []
        while list_of_lists:
            current = set(list_of_lists.pop(0))
            i = 0
            while i < len(list_of_lists):
                if current.intersection(set(list_of_lists[i])):
                    current.update(list_of_lists[i])
                    list_of_lists.pop(i)
                else:
                    i += 1
            merged.append(list(current))
        return merged

    all_outputs = []

    # Collect all 'output' lists from each timestamp
    for timestamp, content in data_to_save.items():
        all_outputs.extend(content.get('output', []))

    # Merge lists with common elements
    merged_outputs = merge_lists_with_common_elements(all_outputs)

    # Print or use the merged_outputs as needed
    # File path to save the new JSON data
    file_path_eva = path+ '/golden_eva.json' 
    # Write the merged output data to the file
    with open(file_path_eva, 'w') as file:
        json.dump(merged_outputs, file, indent=4)  # Use json.dump() to write to the file
    return jsonify({"message": "File saved successfully"})

# 5 Save merged data
@app.route('/save-json', methods=['POST'])
def save_json():
    data = request.json
    for key, attributes in data["theme_attributes"].items():
        if "merged" in attributes and isinstance(attributes["merged"], list):
            # Extract only the second item from each sublist
            attributes["merged"] = [pair[0] for pair in attributes["merged"] if isinstance(pair, list) and len(pair) > 1]

    # Current timestamp
    current_time = datetime.now()

    # Formatting the date and time in the desired format (day/month/year-hour:minute:second)
    formatted_time = current_time.strftime("%d-%m-%Y-%H:%M:%S")

    path=os.getcwd()+ '/golden_data'
    file_path = path+ '/'+formatted_time+'_merged.json'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file_path, 'w') as file:
        json.dump(data, file)
    return jsonify({"message": "File saved successfully"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)