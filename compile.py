import os
import subprocess
import sys

def main():
    command = "g++ -O3 -ffast-math -o sequencial sequencial.cpp"
    print(f"Executando: {command}")
    
    # Adicionando o mingw64 ao PATH dinamicamente apenas no Windows para o g++ encontrar suas DLLs
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")
    
    try:
        result = subprocess.run(command, check=True, shell=True, env=env)
        print("Compilação concluída com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro durante a compilação: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
