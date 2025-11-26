import subprocess
import os


llama_quantize_path = r"C:\Users\ridwn\Documents\GitHub\llama.cpp\build\bin\Release\llama-quantize.exe"
input_path = r"C:\Users\ridwn\Documents\GitHub\SleepDataAnalysis\Training\gemma-SleepAnalysisDataBF16.gguf"
output_path = r"C:\Users\ridwn\Documents\GitHub\SleepDataAnalysis\Training\gemma-SleepAnalysisDataQ4_K_M.gguf"


subprocess.run([
    llama_quantize_path,
    input_path,
    output_path,
    "Q4_K_M"
])


