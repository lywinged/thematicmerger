import os
import json
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI


def BERT_similarity(titles_list,args):
    threshold=float(args) 
    print(threshold,"threshold")
    def normalize(embeddings):
        # Normalization function (as previously described)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    '''
    The pre-trained BERT is for temporary testing, the final effect requires fine-tune with ground truth data(Such as improving sentiment analysis).
    Model                       score speed size     
    all-mpnet-base-v2           696 570 2800 420
    multi-qa-mpnet-base-dot-v1  666 576 2800 420
    multi-qa-mpnet-base-cos-v1  663 574 2800 420
    all-distilroberta-v1        687 509 4000 290        
    all-MiniLM-L12-v2           687 508 7500 120
    paraphrase-TinyBERT-L6-v2   662 411 4500 240
    paraphrase-albert-small-v2  645 400 5000 43
    '''
    model = SentenceTransformer('all-mpnet-base-v2') 
    
    sentences = [title.lower() for title in titles_list]
    embeddings = model.encode(sentences)  #export TOKENIZERS_PARALLELISM=false 
    embeddings = np.array(embeddings).astype("float32")
    normalized_embeddings = normalize(embeddings)

    # KNN algorithm:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(normalized_embeddings)
    grouped_titles = []
    already_grouped = set()
    top_k=5 #len(titles_list) 
    for i, embedding in enumerate(normalized_embeddings):
        if i not in already_grouped:
            # Search the index
            D, I = index.search(embedding.reshape(1, -1), k=top_k)
            group = []
            for idx, similarity in zip(I[0], D[0]):
                if similarity > threshold and idx != i and idx not in already_grouped:
                    group.append(titles_list[idx])
                    already_grouped.add(idx)

            if group:  # Only add non-empty groups
                group.append(titles_list[i])
                already_grouped.add(i)
                grouped_titles.append(group)
            
    # Sort by initial letter, we can sort by the similarity score as well
    grouped_titles = [sorted(sublist, key=lambda item: item.lower()) for sublist in grouped_titles]
    return grouped_titles
 

def GPT_similarity(titles_list, args,max_retries=3):
    key=os.getenv('OPENAI_API_KEY')  # export OPENAI_API_KEY="" 
    client = OpenAI(
        api_key=key,
    ) 
        
    grouped_titles = []
    messages=[]
    print(args,"-GPT args")
    if (args is not None and "humanfeed" in args):
        file_path = os.path.join('golden_data', 'golden_hf.json')
        # Read and parse the JSON file
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                latest_timestamp = max(data.keys())
                latest_output = data[latest_timestamp]["output"]
                latest_input = data[latest_timestamp]["input"]
                # Print or process the last output
                if latest_output is not None:
                    hfdata_in=latest_input
                    hfdata_out=latest_output
                    print(latest_output)
                else:
                    print("No 'output' key found in the JSON data.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print("Error decoding JSON from the file.")
        except Exception as e:
            print(f"An error occurred: {e}")
        parts = args.split("_humanfeed")
        # Get the first part of the split
        args = parts[0]       
        messages=[ #Provide the last saved human feedback examples for current data.
                {"role": "user", "content": f"{titles_list}, if {args} of above items mean the same thing, list it and only response me the list in list format. Please refer this example: if the input is {hfdata_in}, the output should be {hfdata_out}.Don't reply any natural language."},] 
    elif "topic"in args: 
        messages=[ #generate the new topic list.
                {"role": "user", "content": f"{titles_list}, Generate a similar list in {args}, and only response me the list in list format. Don't reply any natural language."},]
    else:
        messages=[ #The args here can be used to adjust the output. If we put "two", it would only merged one most similar theme.
                {"role": "user", "content": f"{titles_list}, if {args} of above items mean the same thing, list it and only response me the list in list format. Don't reply any natural language."},] 
    
    attempts=0
    while attempts < max_retries:   
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4-1106-preview",
            )
            # Check if the response is not None and has content
            if response and response.choices and response.choices[0].message.content:
                api_response = response.choices[0].message.content
                try:
                    #There are only two list templates from GPT4, formal list and html list.
                    if "- "in api_response:
                        lines = api_response.replace("'''", "").strip().split('\n')
                        processed_lines = [line.strip()[2:] for line in lines]  # Remove '- ' from each line
                        formatted_string = '[' + ', '.join(processed_lines) + ']'
                        api_response = formatted_string
                    else:
                        api_response = api_response.replace("json", "", 1).replace("'''", "").strip() 
                    #print(api_response,type(api_response)) 
                    items = api_response.strip("[]").split(", ")

                    # Function to check and add quotes
                    def add_quotes(item):
                        item = item.strip()
                        if not (item.startswith("'") and item.endswith("'")):
                            return f"'{item}'"
                        return item
                    if "'" not in api_response: 
                        # Apply the function to each item
                        formatted_items = [add_quotes(item) for item in items]
                        # Reassemble into a string
                        api_response = "[" + ", ".join(formatted_items) + "]"
                    #print(api_response,type(api_response))
                    api_response = ast.literal_eval(api_response)
                   
                    # Convert the tuples back to lists
                    if "topic"in args:
                        grouped_titles=api_response
                    else: 
                        grouped_titles=GPT_unique_groups_and_sort(api_response)
                        
                    return grouped_titles

                except json.JSONDecodeError as json_error:
                    print(f"JSON parsing error: {json_error}")
                    attempts += 1
            else:
                print("No response or empty content received from the API")
                attempts += 1
                
        except Exception as e:
            print(f"Error in GPT-4 API call: {e}")
            attempts += 1
          
    return grouped_titles
   

# Term frequency, it's for testing the pipeline
def TF_similarity(titles_list,args):
    args=int(args)
    def are_similar(title1, title2,args):
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        common_words_count = sum(1 for word in words1 if word in words2)
        return common_words_count >= args 
    grouped_titles = []
 
    for title in titles_list:
        found_group = False
        # Check if the title is already in a group
        for group in grouped_titles:
            if title in group:
                found_group = True
                break
        # If not in a group, check for similarity and add to a new or existing group
        if not found_group:
            similar_group = None
            for group in grouped_titles:
                if any(are_similar(title, existing_title,args) for existing_title in group):
                    similar_group = group
                    break
            if similar_group:
                similar_group.append(title)
            else:
                grouped_titles.append([title])
    # Filtering out sublists with only one item
    grouped_titles = [group for group in grouped_titles if len(group) > 1]
    return grouped_titles



def GPT_unique_groups_and_sort(grouped_titles):
    # Create a set for each group
    groups = [set(group) for group in grouped_titles]

    # Iterate over the groups and merge groups with common elements
    merged_groups = []
    while groups:
        current_group = groups.pop(0)
        indices_to_merge = []

        for i, group in enumerate(groups):
            if current_group.intersection(group):
                indices_to_merge.append(i)

        # Merge groups and remove merged groups from the main list
        for i in sorted(indices_to_merge, reverse=True):
            current_group.update(groups.pop(i))

        merged_groups.append(current_group)

    # Sort each merged group and convert back to list
    sorted_merged_groups = [sorted(group, key=lambda x: x.lower()) for group in merged_groups]

    return sorted_merged_groups

