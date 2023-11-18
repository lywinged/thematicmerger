import os
import json

## Evaluation
'''
    Input_data: All the titles value 
    Output_data: The output data by model
    Eva_data: The data is a list with sevaral grouped sublist that we labeled before. We save the ground truth data in './golden_data/golden_eva.json' 
The Eva method:
    1. If there more than two items in both Eva_data's sublist and Input_data list, but the output_data DID NOT group them together, it's False.
    2. If the output_data grouped some of them, but miss one, it's False.
    3. If the output_data grouped them correctly, it's True.
'''

def Eva(input_data,output_data):
    Eva_groups={}
    path=os.getcwd()+ '/golden_data'
    file_path_eva = path+ '/golden_eva.json' 
    with open(file_path_eva, 'r') as file:
        Eva_groups = json.load(file)
            
    # Convert eva groups to a dictionary for easy lookup
    eva_dict = {item: group for group in Eva_groups for item in group}
    #print(eva_dict)
    # Convert output data to a dictionary format
    output_dict = {}
    for group in output_data:
        for item in group:
            output_dict[item] = group

    analysis_results = []
    processed_groups = set()

    # Initialize counters
    count_true = 0
    count_miss_all = 0
    count_no_miss_all = 0
    count_miss_and_extra = 0

    for item in input_data:
        # Convert the group list to a tuple for hashing
        eva_group_key = tuple(eva_dict[item]) if item in eva_dict else None

        if eva_group_key and eva_group_key not in processed_groups:
            Eva_group = set(eva_dict[item])
            processed_groups.add(eva_group_key)  # Mark this group as processed

            # Initialize sets for missing and extra items
            eva_not_in_output = set(Eva_group)
            output_not_in_eva = set()

            all_grouped_correctly = True
            partial_grouping = False
            for group_item in Eva_group:
                output_group = set(output_dict.get(group_item, []))
                if output_group != Eva_group:
                    all_grouped_correctly = False
                    eva_not_in_output -= output_group
                    output_not_in_eva.update(output_group.difference(Eva_group))
                    # Check for partial grouping
                    if output_group.intersection(Eva_group):
                        partial_grouping = True

            if eva_not_in_output and output_not_in_eva:
                count_miss_and_extra += 1  # There are both missing and extra items
            if all_grouped_correctly:
                eva_not_in_output.clear()
                count_true += 1
            elif partial_grouping:
                count_no_miss_all += 1  # Some items are correctly grouped, but not all
            else:
                count_miss_all += 1  # The group is completely missing

            analysis_results.append({
                'Eva_group': Eva_group,
                'correctly_grouped': all_grouped_correctly,
                'missing_in_output': eva_not_in_output,
                'extra_in_output': output_not_in_eva
            })

    # Display results
    # for result in analysis_results:
    #     results_str += (f"Item: {result['Eva_group']}\n"
    #                     f"Correctly Grouped: {result['correctly_grouped']}\n"
    #                     f"missing_in_output: {result['missing_in_output']}\n"
    #                     f"extra_in_output: {result['extra_in_output']}\n"
    #                     "-----\n")
    
    print(f"Total True: {count_true}, Total Miss All: {count_miss_all},\n "
      f"Total Not Miss All: {count_no_miss_all}, Total Miss and Extra: {count_miss_and_extra}")
 
    return {
        'counts': {
            'Total True': count_true,
            'Total Miss All': count_miss_all,
            'Total Partially Correct': count_no_miss_all,
            'Total Miss and Extra': count_miss_and_extra
        },
        'analysis_results': analysis_results
    }