import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch 

# This is code in which I tried to use a local LLM but they are just so poor that I cant get a single inteligible output from them
# I highly recommend you just modify the code to use the direct user response instead since none of the open source models I tried (including Mixtra-8x7b) ever outputted a reasonable response
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")#attn_implementation="flash_attention_2", 

def interpret_response(response, options):
    prompt = f"Given the response '{response}', which of the following options is most likely chosen? Do not respond with anything else other than exactly the text of the option. Options: {', '.join(options)}."
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=256)

    input_ids_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    generated_ids = outputs[0, input_ids_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    selected_option = None
    for option in options:
        if option in output_text:
            selected_option = option
            break
    
    return selected_option



dict = json.load(open("workflow.json"))

# Create a hashable dict to lookup the data based on IDs so we only need to traverse nodes list once
nodes = {data['id']: data for data in dict['nodes']}

# This is just assigned by reference so really just for readability
edges = dict['edges']

# Create a nested dictionary structure to represent the DAG. This allows us to efficently determine which possible edged are available from any given node by just traversing the edge list once
edge_dag = {}
for edge in edges:
    
    # Ensure the dict key exists since we will be modifying the nested dict directly
    if edge['source'] not in edge_dag:
        edge_dag[edge['source']] = {}
    sourceHandle = edge['sourceHandle'][1:]
    edge_dag[edge['source']][sourceHandle] = edge['target']
    

# This will traverse the dag starting from the first edge listed in edges
current_node_id = edges[0]['source']
while True:
    cur_node = nodes[current_node_id]['data']
    
    print(cur_node['description'])
    if cur_node['type'] == "end":
        break
    
    available_options = list(edge_dag[current_node_id].keys())
    user_response = input("Type your response: \n")
    
    # Use the LLM to interpret the user's response and choose an option
    chosen_option = interpret_response(user_response, available_options)
    if chosen_option in available_options:
        current_node_id = edge_dag[current_node_id][chosen_option]
    else:
        print("I'm sorry, I couldn't understand your response. Please try again.")
