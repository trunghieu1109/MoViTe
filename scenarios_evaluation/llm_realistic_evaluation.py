import json
import os
import time

from openai import OpenAI


class LLMAPI(object):

    def __init__(self):
        self.total_time_step = 6

    def openai_gpt_4(self, model, messages, temperature):
        
        api_key_path = "E:/Exercise/NCKH/openai-api-key.txt"
        
        api_key = ""
        
        with open(api_key_path, 'r') as file:
            api_key = file.read()

        client = OpenAI(api_key=api_key)  # please add your api_key here

        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=temperature,
        )
        create_time = time.time() - start_time
        middle_time = time.time()
        output = completion.choices[0].message.content
        output_time = time.time() - middle_time
        total_time = time.time() - start_time
        return output, create_time, output_time, total_time


    def evaluate_R_MR_extra_4full_experiments(self, model, repeat_time):
        
        path_prefix = "./"

        road_list = ["road1", "road2", "road3", "road4", "road33"]
        road_list = ["road33"]

        scenarios_description = "In our experiment, a scenario is created by incrementally introducing dynamic obstacles, such as vehicles and pedestrians, throughout the driving duration."

        road_description = {
            "road1": scenarios_description + "In the following scenario, Ego's driving intention is to first drive on a dual-way road with four lanes, then turn right to a one-way road of four lanes. At the next junction, Ego intends to turn left onto a one-way road, drive a short distance, and then stop.\n",
            "road2": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a four-lane, two-way road, then turn right at a T-intersection. After a short distance, Ego will turn left onto a one-way road.\n",
            "road3": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a six-lane, two-way road, pass one intersection, and then turn left at the second intersection.\n",
            "road4": scenarios_description + "In the following scenario, Ego's driving intention is to first drive on a straight, two-lane road, pass one intersection, turn left, and then make another left at the end of the road, where it connects to a curved boulevard.\n",
            "road33": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a six-lane, two-way road, pass one intersection, and then turn left at the second intersection.\n",}
        realistic_counts = {road: 0 for road in road_list}
        realism_scores = {road: [] for road in road_list}

        for road in road_list:
            # Specify the path to the scenario JSON file
            randomly_select_scenarios_path = path_prefix + road + "-scenarios/"
            scenarios_names_list = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                    os.path.isfile(os.path.join(randomly_select_scenarios_path, f))]

            for scenario_name in scenarios_names_list:
                collided = False
                randomly_select_scenario_path = randomly_select_scenarios_path + scenario_name
                print(randomly_select_scenario_path)
                with open(randomly_select_scenario_path, 'r') as json_file:
                    randomly_select_scenario = json.load(json_file)
                self.total_time_step = len(randomly_select_scenario.keys())-1
                object_refs = set()
                for timestep in randomly_select_scenario.keys():
                    object_refs.update(randomly_select_scenario[timestep].keys())
                object_refs = sorted(object_refs)
                params_all = ""
                for time_step in range(self.total_time_step + 1):
                    if collided:
                        break

                    junction_info = None
                    lane_info = None
                    weather_info = None
                    time_info = None

                    params = ""
                    for object_ref in object_refs:
                        if f'timestep_{time_step}' in randomly_select_scenario and object_ref in randomly_select_scenario[f'timestep_{time_step}']:
                            configuration = randomly_select_scenario[f'timestep_{time_step}'][object_ref]
                            if configuration is not None: 
                                if 'position' in configuration:
                                    for parameter_object in ["position", "rotation", "velocity", "angular_velocity"]:
                                        params += f"The '{parameter_object}' of {object_ref} is ({configuration[parameter_object]['x']}, {configuration[parameter_object]['y']}, {configuration[parameter_object]['z']}).\n"

                                    if 'type' in configuration:
                                        params += f"The {object_ref} is a {configuration['type']}"
                                    
                                    if 'Collided_With_Ego' in configuration:
                                        if configuration['Collided_With_Ego'] == True:
                                            params += " and it collided with the Ego.\n"
                                            collided = True
                                        else:
                                            params += ".\n"
                                elif "Left_Boundary" in configuration:
                                    lane_info = configuration
                                elif "rain_level" in configuration:
                                    weather_info = configuration
                                elif "time" in configuration:
                                    time_info = configuration
                                else:
                                    if object_ref == "Ahead_Traffic_Signal":
                                        continue
                                    
                                    junction_info = configuration
                            else:
                                params += f"The '{object_ref}' is not present at {round(time_step * 0.5, 1)} seconds.\n"
                                
                    params += f"Rain level is {weather_info['rain_level']}"
                    params += f"Fog level is {weather_info['fog_level']}"
                    params += f"Wetness level is {weather_info['wetness_level']}"
                    
                    params += f"Time of day is {time_info['time']}"

                    params += f"Left Boundary is {lane_info['Left_Boundary']}.\n"
                    params += f"Right Boundary is {lane_info['Right_Boundary']}.\n"
                    params += f"Left Lane is in the {lane_info['Left_Lane_Direction']}.\n"
                    params += f"Right Lane is in the {lane_info['Right_Lane_Direction']}.\n"

                    if junction_info['Junction_Position'] != "out of":
                        params += f"Ego is {junction_info['Junction_Position']} the junction.\n"
                        if junction_info["Has_Horizontal_Left_Entry"] == True:
                            params += "This junction has horizontal left entry.\n"
                        if junction_info["Has_Horizontal_Right_Entry"] == True:
                            params += "This junction has horizontal right entry.\n"
                        if junction_info["Has_Vertical_Inverse_Entry"] == True:
                            params += "This junction has vertical inverse entry.\n"
                        if junction_info["Has_Vertical_Forward_Entry"] == True:
                            params += "This junction has vertical forward entry.\n"
                                    
                        else:
                            params += "Ego is not in the junction.\n"
                    params_all += f"At {round(time_step * 0.5, 1)} seconds:\n{params}\n"

                messages = []
                sys_message = {}
                sys_message["role"] = "system"
                sys_message["content"] = """You know a lot about autonomous driving and some realistic autonomous driving crash datasets.
You are a helpful crash scenario realism evaluation assistant.
You will evaluate the following autonomous driving crash scenario, and check whether the scenario is realistic and natural. A collision is normal and you will evaluate realism of this case.
You will provide the corresponding realism score. The scale is 1.0-10.0, where 1.0 is unrealistic and 10.0 is realistic.
You will also provide the Probability and Confidence of the outputs as a percentage. Probability refers to the likelihood of your output, while Confidence indicates the certainty in the prediction for your output.
You know the scenario is from the LGSVL simulator."""
                messages.append(sys_message)
                prompt = {}
                prompt["role"] = "user"
                prompt[
                    "content"] = f"""{road_description[road]} This is a scenario generated for testing the ego vehicle. 
                    The scenario starts at 0.0 seconds, with all objects starting from rest. {params_all} The scenario lasts for {self.total_time_step} time steps, each lasting 0.5 seconds.

Your task is to evaluate the degree of realism of this scenario on a scale from 1 to 10 based on the following factors:
- Physical and Environmental Accuracy
- Behavioral Realism of Other Road Users
- Dynamic Scenario Elements
- Edge Case Representation
- Temporal Consistency and Detail
For each factor, provide the realism score (1-10) and a brief reason for the score. After evaluating each factor, calculate the **Average Realism Score** (average of all individual scores) and determine if the scenario is **Realistic** (True or False) based on the overall evaluation.

Please provide the output in the following JSON format:

```json
{{
  "Physical and Environmental Accuracy": {{
    "realism_score": <score>,
    "reason": "<brief reason>"
  }},
  "Behavioral Realism of Other Road Users": {{
    "realism_score": <score>,
    "reason": "<brief reason>"
  }},
  "Dynamic Scenario Elements": {{
    "realism_score": <score>,
    "reason": "<brief reason>"
  }},
  "Edge Case Representation": {{
    "realism_score": <score>,
    "reason": "<brief reason>"
  }},
  "Temporal Consistency and Detail": {{
    "realism_score": <score>,
    "reason": "<brief reason>"
  }},
  "Average Realism Score": <average score>,
  "Realistic": <True or False>
}}"""
                messages.append(prompt)

                response = OpenAI.ChatCompletion.create(
                    model="gpt-4",  
                    messages=messages
                )

                first_chunk_response = response['choices'][0]['message']['content']

                for i in range(1, len(chunks)):
                    chunk = chunks[i]
                    prompt_update = {
                        "role": "user",
                        "content": f"Continuing from the previous details: {chunk}"
                    }
                    messages.append(prompt_update)
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-4",  
                        messages=messages
                    )
                    
                    chunk_response = response['choices'][0]['message']['content']
                final_prompt = {
                    "role": "user",
                    "content": "Now that you've processed all the chunks, please evaluate the scenario based on the details provided."
                }

                messages.append(final_prompt)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages
                )
                # final_evaluation = response['choices'][0]['message']['content']


                for index in range(repeat_time):
                    output, create_time, output_time, total_time = self.openai_gpt_4(model, messages, 0)

                    #save 
                    evaluate_R_MR_extra_4full_experiments_results_path = "./outputs_results/" + road + "-scenarios/" + scenario_name + "/"
                    if not os.path.exists(evaluate_R_MR_extra_4full_experiments_results_path):
                        os.makedirs(evaluate_R_MR_extra_4full_experiments_results_path)
                    evaluate_R_MR_extra_4full_experiments_results_file_name = evaluate_R_MR_extra_4full_experiments_results_path + scenario_name + \
                                                                                model.split("/")[
                                                                                    -1] + "_" + str(
                        index) + ".txt"
                    with open(evaluate_R_MR_extra_4full_experiments_results_file_name, 'w') as file:
                        file.write(f"model: {model}\n\n")
                        file.write(messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\n\n")
                        file.write(output + "\n\n\n")
                        file.write(
                            f"create_time: {create_time}s output_time: {output_time}s total_time: {total_time}s\n")


                    print(
                        road + " " + scenario_name + "===================================================")
                    print(f"model: {model}")
                    print(f"index: {index}\n")

                    #stats
                    try:
                        json_start_index = output.rfind('{')
                        if json_start_index == -1:
                            raise ValueError("No valid JSON found in the output.")
                        json_string = output
                        json_string = json_string[7:-3].strip()
                        print(json_string)
                        output_data = json.loads(json_string)                                                
                        average_score = output_data.get("Average Realism Score", "N/A")
                        is_realistic = output_data.get("Realistic", "N/A")
                        if is_realistic:
                            realistic_counts[road] += 1
                        realism_scores[road].append(average_score)
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON output:", e)

        print("Summary of Realistic Scenarios:")
        for road in road_list:
            scenarios_names_list = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                    os.path.isfile(os.path.join(randomly_select_scenarios_path, f))]
            total = len(scenarios_names_list)                                    
            count = realistic_counts[road]
            avg_score = sum(realism_scores[road]) / total
            print(f"{road}:\nRealistic scenarios: {count}")
            print(f"Average Realism Score: {avg_score:.2f}")
            print(f"Total scenarios: {total}")


if __name__ == '__main__':
    llmapi = LLMAPI()

    model = "gpt-4o"
    repeat_time = 1

    llmapi.evaluate_R_MR_extra_4full_experiments(model, repeat_time)
