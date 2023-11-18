import pytest
from flask_testing import TestCase
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
import json
from datetime import datetime
from app import app as flask_app
from app import merging_titles
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

@pytest.fixture
def app():
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

# Total 6 test cases

#1 Test for home/ route: Index.html
'''
    Input:  Send get request, check the response code.
    Output: Check if the h1 title contains "Thematic"
'''
def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Thematic' in response.data.decode() 


#2 Test for '/GPTthemes': Process themes merging 
'''
    Input:  Send test data and choose TF( Term frequency) model. 
    Output: Check response, it should merge  "conveni_easi_use to "app_easi_use".
'''
def test_GPTthemes_process(client):
    # First POST request with form data
    form_data = {
        'function-select': 'TF',
        'args-input': '2',
        'gen-input': ''
    }
    response_1 = client.post('/GPTthemes', data=form_data)
    assert response_1.status_code == 200
    assert "First part of data received" in response_1.get_json().get("message", "")

    # Second POST request with JSON data
    json_data = {
        "theme_attributes": {
            "absolutely_terrible": {},
            "access_account_info": {},
            "access_easi": {},
            "account_activity": {},
            "account_balance": {},
            "account_bank": {},
            "account_inform": {},
            "account_number": {},
            "app_authent": {},
            "app_conveni": {},
            "app_easi": {},
            "app_easi_use": {},
            "conveni_easi_use":{}
        },
        "titles": {
            "absolutely_terrible": "absolutely terrible experience",
            "access_account_info": "access account info",
            "access_easi": "easy access",
            "account_activity": "account activity",
            "account_balance": "account balance",
            "account_bank": "bank account",
            "account_inform": "account information",
            "account_number": "account number",
            "app_authent": "login to the app",
            "app_conveni": "convenient app",
            "app_easi": "app is easy",
            "app_easi_use": "easy to use",
            "conveni_easi_use": "convenient and easy to use"
        }
    }
    response_2 = client.post('/GPTthemes', json=json_data)
    assert response_2.status_code == 200
    # Parse the JSON response
    response_json = response_2.json

    # Assertions to validate the JSON response content
    expected_length=2
    assert 'theme_attributes' in response_json
    assert len(response_json['theme_attributes']['app_easi_use']['merged']) == expected_length
    
    
#3 Test for /generateData: GPT generate new data
'''
    Input:  Send test data to GPT, let it generate a similar dataset.
    Output: It should generate a new data then send back.
'''
def test_generate_data(client):
 # First POST request with form data
    form_data = {
        'function-select': 'GPT',
        'args-input': 'some',
        'gen-input': 'Bank app'
    }
    response_1 = client.post('/GPTthemes', data=form_data)
    assert response_1.status_code == 200
    assert "First part of data received" in response_1.get_json().get("message", "")

    # Second POST request with JSON data
    json_data = {
        "theme_attributes": {
            "absolutely_terrible": {},
            "access_account_info": {},
            "access_easi": {},
            "account_activity": {},
            "account_balance": {},
            "account_bank": {},
            "account_inform": {},
            "account_number": {},
            "app_authent": {},
            "app_conveni": {},
            "app_easi": {},
            "app_easi_use": {},
            "conveni_easi_use":{}
        },
        "titles": {
            "absolutely_terrible": "absolutely terrible experience",
            "access_account_info": "access account info",
            "access_easi": "easy access",
            "account_activity": "account activity",
            "account_balance": "account balance",
            "account_bank": "bank account",
            "account_inform": "account information",
            "account_number": "account number",
            "app_authent": "login to the app",
            "app_conveni": "convenient app",
            "app_easi": "app is easy",
            "app_easi_use": "easy to use",
            "conveni_easi_use": "convenient and easy to use"
        }
    }
    response_2 = client.post('/GPTthemes', json=json_data)
    assert response_2.status_code == 200
    # Parse the JSON response
    response_json = response_2.json
    assert 'theme_attributes' in response_json
    assert 'titles' in response_json
    

#4 Test for '/save-json' endpoint: save the _merged.json
'''
    Input:  Send test data with current time and save to golden_data/datetime_merged.json.
    Output: 1. Get response without error
            2. Read file and verify the titles value in test_data.
'''
def test_save_json(client):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y-%H:%M:%S")
    test_data = {
        'theme_attributes': {},  
        'titles': {'time': formatted_time} 
    }
    response = client.post('/save-json', json=test_data)
    assert response.status_code == 200
    file_path_eva= "./golden_data/"+formatted_time+"_merged.json"
    with open(file_path_eva, 'r') as file:
        saved_data = json.load(file)
        assert 'time' in saved_data['titles']
        assert saved_data['titles']['time'] == formatted_time
        
#5 Test for '/save-hf-json' endpoint: save golden_hf.json
'''
    Input:  Send testdata with current time and save to golden_data/golden_hf.json.
    Output: 1. Get response without error
            2. Read file and verify the titles value in test_data.
    Note: The port also save golden_eva.json, it depends on merge_lists_with_common_elements function,
'''
def test_save_hf_json(client):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d/%m/%Y-%H:%M:%S")
    test_data = {
        'theme_attributes': {},  
        'titles': {'time': formatted_time} 
    }
    response = client.post('/save-hf-json', json=test_data)
    assert response.status_code == 200
    # Check if the data is saved correctly
    file_path_eva= "./golden_data/golden_hf.json"
    with open(file_path_eva, 'r') as file:
        saved_data = json.load(file)
        assert 'time' in saved_data[formatted_time]['titles']
        assert saved_data[formatted_time]['titles']['time'] == formatted_time


#6 Test merging_titles function   
'''
    Input:  Original data and grouped data
    Output: Merge the grouped data to original data. (The merged data is not the final theme_attributes)
'''
def test_merging_titles():
    data = {
            "theme_attributes": {
                "app_conveni": {},
                "app_easi": {},
                "app_easi_use": {},
                "conveni_easi_use":{}
            },
            "titles": {
                "app_conveni": "convenient app",
                "app_easi": "app is easy",
                "app_easi_use": "easy to use",
                "conveni_easi_use": "convenient and easy to use"
            }
        }
    grouped_titles=[['app is easy','convenient and easy to use']]
    expected_data={
        'app_conveni': {}, 
        'app_easi': {'merged': [['app_easi', 'app is easy'], ['conveni_easi_use', 'convenient and easy to use']]}, 
        'app_easi_use': {}, 
        'conveni_easi_use': {}
        }
    assert merging_titles(data,grouped_titles) == expected_data
