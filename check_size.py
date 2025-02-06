"""
sole purpose is checking the fucking json file schema
"""
import torch
import os 
import json 


# with open("./models/voices.json", "r") as jsondata:
#     data = json.load(jsondata)
shidd = torch.load("./models/af_heart.pt", weights_only=True).numpy()
# print(shidd)
print(shidd)


# print(data["af"])
# print(type(data["af"]))
# for name in data.keys():
#     print(name)
