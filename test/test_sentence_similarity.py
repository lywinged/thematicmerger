import json 
from unittest.mock import mock_open, patch
from sentence_similarity import BERT_similarity,GPT_similarity,TF_similarity,GPT_unique_groups_and_sort 

# Total 6 test cases

titles_list= [
            "absolutely terrible experience",
            "access account info",
            "easy access",
            "account activity",
            "account balance",
            "bank account",
            "account information",
            "account number",
            "login to the app",
            "convenient app",
            "app is easy",
            "easy to use",
            "app improvements",
            "log into the app",
            "open the app",
            "authentication / login",
            "works fine",
            "biometric / finger print",
            "cash checks",
            "change password",
            "mobile check deposit",
            "mobile check",
            "close my account",
            "convenient",
            "convenient and easy to use",
            "great customer",
            "great customer service",
            "mobile deposit",
            "access code",
            "great experience",
            "great tool",
            "friendly service",
            "friendly staff",
            "great service",
            "horrible service",
            "ional service",
            "locked out of my account",
            "login info",
            "login issue",
            "manage my account",
            "cannot access my account",
            "cannot login",
            "open an account",
            "password manager",
            "polite n professional",
            "security",
            "useful",
            "view my account",
            "waste of time",
            "worst company"
        ]

expected_BERT=[['access account info', 'account information'], ['log into the app', 'login to the app'], ['convenient and easy to use', 'easy to use'], ['mobile check deposit', 'mobile deposit'], ['great customer service', 'great service'], ['friendly service', 'friendly staff'], ['cannot login', 'login issue']]
expected_TF= [['access account info', 'cannot access my account', 'view my account'], ['login to the app', 'log into the app', 'open the app'], ['easy to use', 'convenient and easy to use'], ['mobile check deposit', 'mobile check', 'mobile deposit'], ['close my account', 'locked out of my account', 'manage my account'], ['great customer', 'great customer service', 'great service']]

# 7  test the BERT similarity 
'''
    Input:  Send the titles value in a list format.
    Output: the grouped list with similar items.
'''
def test_BERT_similarity():
    global titles_list,expected_BERT
    args = "0.75"  
    assert BERT_similarity(titles_list, args) == expected_BERT

# 8  test the Term Frequency similarity 
'''
    Input:  Send the titles value in a list format.
    Output: the grouped list with similar items.
'''    
def test_TF_similarity():
    global titles_list,expected_TF
    args = "2"  
    assert TF_similarity(titles_list, args) == expected_TF

# 9  test the GPT similarity 
'''
    Input:  Send the titles value in a list format.
    Output: the grouped list with similar items.
'''    
def test_GPT_similarity():
    global titles_list
    args = "two"
    assert len(GPT_similarity(titles_list, args)) > 5

# 10  test the GPT data generation 
'''
    Input:  Send the titles value in a list format.
    Output: let GPT generates the similar data list.
'''    
def test_data_generation_in_GPT_similarity():
    global titles_list
    args = "Bank App topic"
    assert len(GPT_similarity(titles_list, args)) >= 20 

# 11  test the GPT similarity with human feedback examples 
'''
    Input:  Send the titles value in a list format.
    Output: the grouped list with similar items.
'''
def test_human_feedback_in_GPT_similarity():
    global titles_list
    args = "humanfeed"
    hfdata_in = titles_list 
    hfdata_out = [
        ["absolutely terrible experience", "horrible service"],
        ["account information", "access account info"],
        ["easy access", "convenient and easy to use", "convenient app", "app is easy", "easy to use"],
        ["login to the app", "open the app"],
        ["mobile check", "mobile check deposit", "mobile deposit"],
        ["polite n professional", "great customer service", "great service", "friendly staff", "friendly service"],
        ["locked out of my account", "cannot access my account", "cannot login"]
    ]
    # Mock data setup
    mock_file_content = json.dumps({
        "2023-11-11-11:11:11": {
            "input": hfdata_in,
            "output": hfdata_out
        }
    })
    mocked_open = mock_open(read_data=mock_file_content)
        # Patch the open function in the Eva module
    with patch('builtins.open', mocked_open):
        # Call the Eva function
        results = GPT_similarity(titles_list, args)
    assert len(results) > 5   

# 12  test GPT_unique_groups_and_sort
'''
    Input:  Send a grouped list with comman item in multiple group.
    Output: Output merge them to an unique group.
'''
def test_GPT_unique_groups_and_sort():
    grouped_titles=[
            [
                "access account info",
                "account information",
            ],
            [
                "easy access",
                "easy to use",
            ],
            [
                "easy access",
                "app is easy"
            ]
        ]
    unique_groups=[
        [
            'access account info', 
            'account information'],
        [
            'app is easy', 
            'easy access', 
            'easy to use'
        ]
    ]
    assert GPT_unique_groups_and_sort(grouped_titles) == unique_groups