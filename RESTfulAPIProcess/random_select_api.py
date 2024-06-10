import json
import requests
import random


def get_action_space():
    json_file = open('RESTful_API.json', 'r')
    content = json_file.read()
    restful_api = json.loads(s=content)
    return restful_api


# Execute action
def execute_action(api_id):
    api = action_space[str(api_id)]
    requests.post(api)


requests.post("http://localhost:8933/LGSVL/LoadScene?scene=5d272540-f689-4355-83c7-03bf11b6865f")
action_space = get_action_space()['command']
action_space_size = action_space['num']
print(action_space)

for i in range(1000):
    action_id = random.randint(0, action_space_size - 1)
    execute_action(action_id)
#     收集图像
