import json

def write_data(file:str, data) -> None:
	with open(file, "w", encoding="utf-8") as write_file:
		json.dump(data, write_file, ensure_ascii=False, indent=4)

def read_data(file:str):
	with open(file, "r", encoding='utf-8') as read_file:
		data = json.load(read_file)
	return data

