
import torch
from torch.optim import Adam
from model import OptimusPrime
from data import Bumblebee, collate_batch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

OPTIMUS = f'''
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀⠀⣤⣤⣤⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⣠⡶⢿⡇⢿⣿⡏⢳⣦⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⡛⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⣼⣿⣴⣋⡽⠮⠿⢭⣟⣏⣷⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣧⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡼⣇⣿⡿⠶⣶⣿⣟⡛⣷⣿⢠⠙⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡈⣏⠇⢹⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⢹⠁⣿⠋⠉⢹⠉⠙⣿⡇⣾⣀⣾⠀⢀⣤⡀⢀⡀⠀⠀⢀⣠⣴⣾⠛⢻⡛⢻⡄⢀⣳⡀⢀⣠⠄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣷⣾⢀⣿⡇⠀⠸⠀⠀⣿⣧⡽⠿⣟⣺⣭⠴⢿⡏⣩⣷⡾⢛⣭⣴⣿⣇⠘⣿⣷⣿⡛⠉⢻⣟⣷⠄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⢿⣟⣿⣿⡦⣶⣪⡭⠿⣚⣫⣭⣽⣶⡄⠀⢸⡇⣿⡙⣿⣿⣿⣿⣿⣿⣆⠹⣿⣿⣷⡀⠀⢿⡉⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣤⣶⣿⠿⠛⣉⣭⣶⣾⣿⠿⠟⠛⠉⠉⢻⠀⢸⣷⣿⣇⢻⡿⣿⣿⣿⣿⠟⠀⠹⣿⣿⠃⠀⠘⣷⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣦⣼⣿⠿⠛⣋⡁⣼⢠⣿⡿⠛⠉⠁⠀⠀⢀⡀⢀⣴⣾⠀⢸⣿⡇⢻⡄⠙⠿⠻⠛⠁⠀⢀⣠⣽⣿⣇⡀⠀⠸⣧⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⠿⣛⣭⣴⡾⠟⠛⣧⣿⢸⡿⠀⠀⠀⠀⣰⣿⣿⣷⣾⣿⣿⠀⢸⡏⣇⢸⣷⡀⠀⢀⣠⣴⣾⠿⠛⣿⢻⣿⣹⡀⠀⢻⣆⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⡟⣦⠀⠀⠀⢀⡿⣵⡿⠛⠉⣡⣶⣤⣄⣿⣯⢸⣇⠀⠀⢠⣾⣿⡿⣿⣿⣿⣿⡿⠀⢸⡇⢻⡼⣿⣷⣶⠿⠛⠉⠀⠀⠀⠸⡇⣿⣿⣧⠀⠘⣿⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⢹⠀⢀⣠⣼⣿⣿⠀⢀⣼⣿⣿⣿⣿⡇⣿⢸⣿⣀⣀⣿⡿⠿⠶⠚⠛⠉⠉⠀⠀⢸⡇⠀⢻⣾⣝⣿⡆⠀⢀⣠⡴⠖⠛⢻⡾⣿⣿⣆⠀⢹⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣇⣼⡾⠟⠋⣿⢻⣇⣤⣌⠻⢿⣿⣿⣿⠃⢿⠀⠉⠉⠁⠀⠀⠀⣀⣤⡤⠶⠶⠒⠚⣻⣷⣄⠈⣿⣿⣿⣿⡞⠉⠀⠀⠀⠀⠀⣿⢿⣿⣾⣋⣽⠇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣹⠏⠀⠀⠀⣿⢿⣿⣿⣯⡴⠾⠛⢋⣡⠶⠛⠛⠋⣉⣉⣉⣙⢻⣿⠀⠀⠀⠀⠀⢠⡟⠀⠈⠻⢦⣈⣿⣿⣧⠀⠀⢀⣠⣴⡾⢿⣿⣿⣿⣿⣿⡀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡟⣿⡟⠀⠀⠀⣿⠈⠋⠉⢀⣠⠴⣛⣩⣤⣶⣞⣭⣿⢿⣿⣿⣻⣼⣿⣆⣀⣤⣤⣴⣿⣄⣠⣶⣦⣀⣙⣿⣿⣿⡶⣿⠟⠋⣁⣶⠟⢻⣽⣿⣿⣿⠇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢠⣿⣇⠀⠀⠀⢹⣠⡴⠖⢻⣷⢫⣿⣿⣿⣯⣿⣟⣿⣿⣭⣽⣿⡿⣿⣿⣿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⠋⠉⣿⠀⢸⣿⣿⣿⣿⣷⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣼⣿⣿⣤⣴⣾⢿⡅⠀⣀⣾⢿⣿⣿⣿⣿⣿⣿⡿⣿⣷⣿⣿⣿⡇⣿⣿⡇⠀⠀⢸⣿⣿⡟⢿⣿⣿⣿⣿⣿⣣⣿⠁⣿⣀⣤⡿⠀⢀⣿⣿⣿⣿⣿⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠻⣿⠛⠉⠀⠈⣿⠛⢽⣿⢻⣿⣿⢿⣿⣿⣿⡇⣿⠿⣶⣶⣚⣧⣿⣿⡇⠀⠀⣸⣿⣿⣿⣄⣈⢿⣿⢿⣷⣿⣿⠀⠉⠉⠀⠀⠀⠘⡇⣿⣿⣿⣿⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⡀⣷⡆⠀⠀⠀⠸⣧⣻⣿⢸⣿⣿⡿⢿⣾⣻⡇⣿⣿⣿⣿⣿⣿⣿⠿⠷⠾⠛⠛⠿⢿⣿⣿⣿⣄⣿⠿⠋⢸⣿⠀⠀⠀⠀⠀⠀⠀⡇⣿⣿⣿⣿⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⡇⣿⡇⠀⠀⠀⠀⣿⣿⣿⡾⢿⣿⣿⣿⣿⡶⠷⠾⠛⠛⠉⠁⢀⣠⠤⠴⠒⡆⢠⠀⢰⡉⠻⣿⣽⡏⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⣿⡿⣿⣿⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣧⣿⠿⢀⣀⣤⣴⣿⣿⣿⡷⠾⠛⠋⠉⢀⣀⣠⠤⠴⠒⠻⡆⢸⠀⠀⢀⡠⠇⠸⡄⠈⣇⠀⠈⡻⢦⡀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⣿⣧⡘⠿⢻⡆
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣆⣿⣿⣿⣿⣿⡿⠛⣉⣀⡀⣠⠴⠒⠋⠉⠁⠀⠀⠀⠀⠀⡇⢸⣠⠴⣫⡄⠀⠀⡇⠀⢹⠀⠀⣿⠦⢿⡀⢸⡇⠀⠀⣀⣤⣤⣿⠀⡇⣿⣿⣿⣆⢸⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⢿⡟⣽⣿⠀⣏⠁⠀⡇⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⠀⡖⣻⠋⠀⠀⠈⢻⠀⢈⡇⠀⠸⡄⠘⣧⢸⡇⠀⢸⣷⣾⣿⠏⠀⡇⣿⣿⣿⣿⢸⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠏⠛⠋⢡⣿⠀⠸⣿⣟⡃⣇⠀⠀⠀⠀⠀⣀⣠⡤⠶⠒⠋⠀⠛⠁⠀⣀⣤⣶⣿⣿⣿⣿⣷⣤⡈⠁⢻⡞⣿⠀⠈⠻⣴⠏⠀⠀⠿⢹⣿⣎⢻⣿⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⡟⠀⠀⢀⡿⣿⠀⠀⠈⠳⡇⠻⠤⠶⠚⠋⠉⠁⠀⠀⠀⠀⠀⣀⣤⣶⣿⣿⣿⣿⣿⠿⠛⠻⣿⣿⣿⣷⣜⣷⣿⠀⠀⢀⣀⣤⣤⣶⣾⣶⣿⣿⠃⢸⡇
⠀⠀⠀⠀⠀⠀⣀⣤⡶⠶⠖⠚⢛⠛⠳⢶⣼⡟⠀⠀⢀⣼⣹⣿⢀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⢀⣀⣠⡤⢤⣾⣿⣿⣿⡿⠿⠛⠉⠹⡇⠀⠀⣿⣿⣟⢿⣿⣿⠹⣶⣿⡿⠛⠻⣏⠀⠉⠉⡛⣿⡿⣾⡇
⠀⠀⠀⢀⣴⠞⠋⢰⡇⢰⣿⢻⢻⢻⢶⣦⠙⣷⡀⠀⣸⢧⠟⢿⣿⣿⣿⣷⣶⣶⣤⣴⣲⡾⠿⠟⠒⠒⠛⡇⠙⣿⠉⠀⢧⠀⠀⠀⠀⣧⠀⠀⢸⣿⣿⡎⣿⠁⢀⣼⣏⢀⣠⣤⣸⣶⠀⠀⣿⣿⣿⠛⠁
⠀⠀⠀⣾⠃⠀⣠⡬⣤⣼⣛⠾⣼⣞⡾⡟⠀⠘⣧⣠⣏⡞⠀⠈⠻⣿⡏⢹⡟⠛⠻⣿⠁⠀⠀⠀⠀⠀⠀⣇⠀⣿⠀⠀⢸⡄⠀⠀⠀⢸⠀⠀⠘⣿⣿⣇⣿⣴⡞⢣⣽⣿⣿⣿⣿⣿⠀⠀⣿⣿⡟⠀⠀
⠀⠀⠀⣿⡶⣿⣿⣸⣿⣿⣿⠿⠷⠾⢽⣅⡲⠶⢻⣿⣼⢁⣠⣤⣶⣿⣿⠘⡇⠀⠀⢻⡆⠀⠀⠀⠀⠀⢀⣸⡀⢹⡇⠀⠈⡇⠀⠀⠀⠈⡇⠀⠀⢿⣿⣿⢹⣿⣤⣿⣿⣿⣿⡿⢿⣟⡀⠀⣿⣿⡇⠀⠀
⠀⠀⠀⠈⠛⠿⢯⣜⣿⠏⠀⠀⠀⢀⡿⣨⣿⣶⣤⣿⣷⣯⣿⣿⣿⣿⣿⠀⡇⠀⠀⠐⡿⣦⣰⣒⣶⣿⣿⣿⣷⣾⣇⠀⠀⢻⠀⠀⠀⠀⢷⠀⠀⢸⣿⣿⣾⣿⣸⣿⡏⢠⠟⣠⣿⣿⣿⣦⡈⢹⡇⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢸⡟⣾⠄⠀⠀⣸⡇⣿⣿⣿⠟⠋⠛⢿⣿⣿⣿⣿⣿⡄⢻⠀⠀⠀⡇⠈⠙⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⢸⡆⠀⠀⠀⢸⡄⠀⠀⣿⣿⣇⣿⠛⠛⠻⣿⣺⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣼⢧⡇⠀⠀⠀⣿⢸⣿⣿⡿⢦⣴⣿⣿⣷⡿⣿⡿⣿⡇⢸⡄⠀⠀⢹⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⣇⠀⠀⠀⠀⣇⠀⠀⢸⣿⣟⢿⡀⠀⠀⠈⠉⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣿⣨⡧⠤⠤⢤⣇⡾⣿⣿⣠⣿⣿⣿⣿⣿⣿⣽⣿⣿⣷⠀⣇⠀⠀⢸⠀⠀⢸⢻⣿⣿⣿⣿⡇⣿⣿⠀⠀⢹⡄⠀⠀⢀⣸⠀⠀⠸⣿⣿⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⡿⣧⣤⠶⠦⣼⣿⣿⣿⡏⠈⣿⣿⢿⣿⣿⣿⣏⠉⢹⣿⡀⢻⠀⠀⠘⡇⠀⠸⡄⠙⢿⣿⣿⠇⣿⣿⡄⠀⠈⠓⠒⠋⠉⠀⠀⠀⠀⢿⠹⣯⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⣿⢃⡏⠀⠀⢻⣿⣿⣽⣿⣦⠘⣿⣿⣿⣿⣿⢻⣿⣾⣿⡇⠘⡇⠀⠀⣇⠀⠀⣇⠀⠀⠙⢿⡇⣿⢸⣧⠀⠀⠀⠀⡴⠒⢶⠀⠀⠀⠘⣆⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⡿⡅⣸⢁⣄⡄⣾⣿⢿⣿⠿⣿⣿⢻⣿⣿⣟⣿⣸⣻⡿⣿⣧⠀⠙⠒⠛⠛⠀⠀⢿⣿⣄⠀⠀⠀⣿⠈⣿⡄⠀⠀⠀⡇⠀⠘⡇⠀⠀⠀⢿⣦⢸⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⣧⡇⣿⣼⣿⠃⣿⣿⣾⣿⣷⣤⡿⠿⢿⣿⣿⣇⣿⡟⠋⠀⣿⡀⠀⣴⠲⡆⠀⠀⠸⣿⣿⣦⠀⠀⢸⡀⢹⣧⠀⠀⠀⣇⠀⠀⢹⠀⠀⠀⠸⣿⡟⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀  ⢽⡿⣷⠏⠛⠿⢠⣿⣿⣿⣿⢿⣯⡇⠀⠀⠈⠁⠀⠀⠀⠀⠀⢸⣇⠀⢻⠀⢳⠀⠀⠀⣿⣿⣿⣷⣾⢸⡇⠈⣿⡀⠀⠀⢸⠀⠀⠈⡇⠀⠀⢀⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
░██████████                                              ░████                                                                    
    ░██                                                 ░██                                                                       
    ░██    ░██░████  ░██████   ░████████   ░███████  ░████████  ░███████  ░██░████ ░█████████████   ░███████  ░██░████  ░███████  
    ░██    ░███           ░██  ░██    ░██ ░██           ░██    ░██    ░██ ░███     ░██   ░██   ░██ ░██    ░██ ░███     ░██        
    ░██    ░██       ░███████  ░██    ░██  ░███████     ░██    ░██    ░██ ░██      ░██   ░██   ░██ ░█████████ ░██       ░███████  
    ░██    ░██      ░██   ░██  ░██    ░██        ░██    ░██    ░██    ░██ ░██      ░██   ░██   ░██ ░██        ░██             ░██ 
    ░██    ░██       ░█████░██ ░██    ░██  ░███████     ░██     ░███████  ░██      ░██   ░██   ░██  ░███████  ░██       ░███████  
                                                                                                                                                                     
'''

import shutil, getpass

# clear terminal
print('\033c', end = '')
width = shutil.get_terminal_size().columns
for line in OPTIMUS.splitlines():
    print(line.center(width))
print()
print('-- press ENTER to continue --'.center(width), end = '', flush = True)  
getpass.getpass('', stream = None)
print('\033c', end = '')
        

EPOCHS = 1
shard_path = Path.cwd().parent / 'data' / 'training' / '1M_train.tsv'

print(f'\n\033[34m [SETUP]\033[0m')
print(f'  [total epochs]: {EPOCHS}')
print(f'  [loading file]: {shard_path.name}')


print(f'  [cuda??]')
if torch.cuda.is_available():
    device = 'cuda'
    print(f'\033[32m   [cuda found]\033[0m')
    print(f'\033[32m   [using GPU]\033[0m')
else:
    device = 'cpu'
    print(f'\033[31m   [cuda NOT found]\033[0m')
    print(f'\033[31m   [using CPU]\033[0m')

print(f'  [building dataset]')
dataset = Bumblebee(shard_path)

print(f'  [building dataloader]')
# -- BUILD MODELS --
dloader = DataLoader(
    dataset,
    batch_size = 1024, 
    shuffle = True,
    collate_fn = collate_batch
)

print(f'  [forging Optimus Prime]')
model = OptimusPrime(
    vocab_size = 257,
    pw_vocab_size = 75,
    pad_id = 74,
    hash_pad_id = 256,
    d_model = 256,
    n_heads = 8,
    num_layers = 4, 
    ff_dim = 512, 
    dropout = 0.1
).to(device)


print(f'  [initializing TensorBoard]')
writer = SummaryWriter(log_dir = 'runs/optimus')
print(f'  [building Adam]')
optimizer = Adam(model.parameters(), lr = 1e-4)
print(f'\033[34m [START]\033[0m')

# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()

    # progress bar for current epoch
    progress = tqdm(dloader, desc = f'Epoch {epoch + 1}/{EPOCHS}', leave = False, unit = ' batch')

    for i, batch in enumerate(progress):
        batch_start_time = time.time()

        # move data to GPU/CPU
        hashes = batch['hash'].to(device)
        pw = batch['password'].to(device)

        # forward pass
        logits = model(hashes, pw)
        loss = model.compute_loss(logits, pw)

        # backward pass
        optimizer.zero_grad()   # reset old gradients
        loss.backward()         # compute new gradients via backprop
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)  # clip grad norms to stabilize training
        optimizer.step()        # update weights

        # accumulate loss
        total_loss += loss.item()
        avg_loss = total_loss / (i + 1) # average over batches processed so far

        # log metrics to tensorboard
        global_step = epoch * len(dloader) + i
        writer.add_scalar('Loss/batch', loss.item(), global_step)
        writer.add_scalar('Loss/epoch_avg', avg_loss, global_step)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Gradient/norm', grad_norm.item(), global_step)
        batch_time = time.time() - batch_start_time    # log batch time
        writer.add_scalar('Time/batch_seconds', batch_time, global_step)
        
        # update progress bar
        progress.set_postfix_str(f'loss = {avg_loss:.4f}, grad_norm = {grad_norm.item():.4f}')

    # close progress bar for this epoch
    progress.close()

    # log epoch-level metrics
    epoch_time = time.time() - epoch_start_time
    final_epoch_loss = total_loss / len(dloader)

    writer.add_scalar('Loss/epoch_final', final_epoch_loss, epoch)
    writer.add_scalar('Time/epoch_minutes', epoch_time / 60, epoch)

    print(f'\n [Epoch {epoch + 1}/{EPOCHS} complete]')
    print(f'  [Final Loss]: {final_epoch_loss:.4f}')
    print(f'  [Time]: {epoch_time / 60:.2f} minutes')

writer.close()
print(f'\n\033[34m [COMPLETE]\033[0m')
print(f'  [view results]: tensorboard --logdir=runs')
print(f'  [dashboard]: http://localhost:6006')
