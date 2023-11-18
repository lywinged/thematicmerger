from evaluation import Eva  
import json 
from unittest.mock import mock_open, patch

# 13 Test for evaluation
'''
    Input_data: All the titles value 
    Output_data: The output data by model
    Eva_data(golden_hf_data): The data is a list with sevaral grouped sublist that we labeled before. We save the ground truth data in './golden_data/golden_eva.json' 
The Eva method:
    1. If there more than two items in both Eva_data's sublist and Input_data list, but the output_data DID NOT group them together, it's False.
    2. If the output_data grouped some of them, but miss one, it's False.
    3. If the output_data grouped them correctly, it's True.
'''

def test_Eva():
    # Mock data setup
    golden_hf_data = [
        ["absolutely terrible experience", "horrible service"],
        ["account information", "access account info"],
        ["easy access", "convenient and easy to use", "convenient app", "app is easy", "easy to use"],
        ["login to the app", "open the app"],
        ["mobile check", "mobile check deposit", "mobile deposit"],
        ["polite n professional", "great customer service", "great service", "friendly staff", "friendly service"],
        ["locked out of my account", "cannot access my account", "cannot login"]
    ]
 
    mock_file_content = json.dumps(golden_hf_data)
    mocked_open = mock_open(read_data=mock_file_content)
        
    input_data= [
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
    output_data= [
            [
                "absolutely terrible experience",
                "horrible service"
            ],
            [
                "waste of time",
                "worst company"
            ],
            [
                "access account info",
                "account information"
            ],
            [
                "easy access",
                "app is easy",
                "easy to use",
                "useful",
                "convenient and easy to use"
            ],
            [
                "cannot access my account",
                "cannot login"
            ]
        ]

    # Patch the open function in the Eva module
    with patch('builtins.open', mocked_open):
        # Call the Eva function
        results = Eva(input_data, output_data)
    # Assertions
    # There are 7 ground truth sets, the output should be: 
    '''
    Item: {'horrible service', 'absolutely terrible experience'}
    Correctly Grouped: True
    missing_in_output: set()
    extra_in_output: set()
    -----
    Item: {'access account info', 'account information'}
    Correctly Grouped: True
    missing_in_output: set()
    extra_in_output: set()
    -----
    Item: {'easy to use', 'convenient and easy to use', 'convenient app', 'app is easy', 'easy access'}
    Correctly Grouped: False
    missing_in_output: {'convenient app'}
    extra_in_output: {'useful'}
    -----
    Item: {'login to the app', 'open the app'}
    Correctly Grouped: False
    missing_in_output: {'login to the app', 'open the app'}
    extra_in_output: set()
    -----
    Item: {'mobile check deposit', 'mobile check', 'mobile deposit'}
    Correctly Grouped: False
    missing_in_output: {'mobile check deposit', 'mobile check', 'mobile deposit'}
    extra_in_output: set()
    -----
    Item: {'polite n professional', 'great customer service', 'friendly service', 'great service', 'friendly staff'}
    Correctly Grouped: False
    missing_in_output: {'great service', 'friendly service', 'great customer service', 'polite n professional', 'friendly staff'}
    extra_in_output: set()
    -----
    Item: {'cannot login', 'locked out of my account', 'cannot access my account'}
    Correctly Grouped: False
    missing_in_output: {'locked out of my account'}
    extra_in_output: set()
    -----
    '''
    # The expected counts should be: 
    expected_counts = {
        'Total True': 2,
        'Total Miss All': 3,
        'Total Partially Correct': 2,
        'Total Miss and Extra': 1
    }
    assert results['counts']['Total True'] == expected_counts['Total True']
    assert results['counts']['Total Miss All'] == expected_counts['Total Miss All']
    assert results['counts']['Total Partially Correct'] == expected_counts['Total Partially Correct']
    assert results['counts']['Total Miss and Extra'] == expected_counts['Total Miss and Extra']
